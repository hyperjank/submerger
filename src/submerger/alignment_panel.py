from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .alignment import (
    AlignmentPackage,
    carry_forward_human_reviews,
    HeuristicAlignmentClient,
    OpenAICompatibleAlignmentClient,
    align_subtitles,
    alignment_artifact_path,
    load_alignment_package,
    review_alignment_segment,
    write_aligned_srt_exports,
    write_alignment_sidecar,
)
from .settings import LLMEndpointSettings, settings_from_provider


class AlignmentWorkerSignals(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)


class AlignmentWorker(QRunnable):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.signals = AlignmentWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            provider = self.config["provider"]
            if provider in {"lmstudio", "openai", "deepseek", "custom"}:
                client = OpenAICompatibleAlignmentClient(
                    model=self.config["model"],
                    base_url=self.config["base_url"],
                    api_key=self.config["api_key"],
                    timeout=float(self.config["timeout"]),
                )
            else:
                client = HeuristicAlignmentClient()

            def progress(batch: int, total: int) -> None:
                self.signals.progress.emit(f"Aligning batch {batch}/{total}...")

            output_prefix = Path(self.config["output_prefix"])
            package = align_subtitles(
                self.config["primary_path"],
                self.config["secondary_path"],
                primary_language=self.config["primary_language"],
                secondary_language=self.config["secondary_language"],
                client=client,
                batch_size=int(self.config["batch_size"]),
                progress=progress,
                cache_path=alignment_artifact_path(output_prefix, ".alignment-cache.json"),
                context_retry=bool(self.config.get("context_retry", True)),
                media_language=self.config["primary_language"],
                repair_target_dialogue=bool(
                    self.config.get("repair_target_dialogue", False)
                ),
                repair_cache_path=alignment_artifact_path(
                    output_prefix,
                    ".repair-cache.json",
                ),
            )
            sidecar_path = alignment_artifact_path(output_prefix, ".alignment.json")
            if sidecar_path.exists():
                try:
                    package = carry_forward_human_reviews(
                        package,
                        load_alignment_package(sidecar_path),
                    )
                except (OSError, ValueError):
                    pass
            write_alignment_sidecar(package, sidecar_path)
            exports = None
            if self.config.get("export_srt"):
                exports = write_aligned_srt_exports(package, output_prefix)
            self.signals.finished.emit(
                {
                    "package": package,
                    "sidecar_path": sidecar_path,
                    "exports": exports,
                }
            )
        except Exception as exc:  # noqa: BLE001 - present pipeline failures in the GUI.
            self.signals.failed.emit(str(exc))


class AlignmentPanel(QWidget):
    run_requested = Signal(dict)
    load_alignment_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.package: AlignmentPackage | None = None
        self.sidecar_path: Path | None = None
        self._visible_segment_ids: list[str] = []

        self.primary_path = QLineEdit()
        self.secondary_path = QLineEdit()
        self.output_prefix = QLineEdit("episode-gui")
        self.provider = QComboBox()
        self.provider.addItems(["heuristic", "lmstudio", "openai", "deepseek", "custom"])
        self.provider.setCurrentText("lmstudio")
        self.model = QLineEdit("qwen3.5-4b")
        self.base_url = QLineEdit("http://192.168.86.113:1234/v1")
        self.api_key = QLineEdit("lm-studio")
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.timeout = QSpinBox()
        self.timeout.setRange(10, 600)
        self.timeout.setValue(180)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 50)
        self.batch_size.setValue(12)
        self.primary_language = QLineEdit("en")
        self.secondary_language = QLineEdit("zh")
        self.export_srt = QCheckBox("Also export retimed SRT files")
        self.export_srt.setChecked(False)
        self.load_when_done = QCheckBox("Load alignment in player when finished")
        self.load_when_done.setChecked(False)
        self.context_retry = QCheckBox(
            "Retry unresolved mappings with surrounding dialogue"
        )
        self.context_retry.setChecked(True)
        self.repair_target_dialogue = QCheckBox(
            "Generate missing or meaningfully divergent other-language dialogue"
        )
        self.repair_target_dialogue.setChecked(False)
        self.repair_target_dialogue.setToolTip(
            "Uses the media-language subtitle as authoritative context. Imported "
            "subtitle text is preserved; generated text is stored separately with provenance."
        )
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        self.primary_browse = QPushButton("Browse")
        self.secondary_browse = QPushButton("Browse")
        self.run_button = QPushButton("Run Alignment")
        self.open_alignment_button = QPushButton("Open Alignment…")
        self.load_button = QPushButton("Load Alignment in Player")
        self.load_button.setEnabled(False)

        self.review_summary = QLabel("Run or open an alignment to review it.")
        self.issues_only = QCheckBox("Show unresolved only")
        self.issues_only.setChecked(True)
        self.review_list = QListWidget()
        self.primary_review = QPlainTextEdit()
        self.primary_review.setReadOnly(True)
        self.primary_review.setPlaceholderText("Primary segment")
        self.candidate_list = QListWidget()
        self.review_status = QLabel("")
        self.review_status.setWordWrap(True)
        self.approve_button = QPushButton("Approve Current Selection")
        self.apply_selection_button = QPushButton("Apply Checked Cues")
        self.save_review_button = QPushButton("Save Alignment")
        for button in (
            self.approve_button,
            self.apply_selection_button,
            self.save_review_button,
        ):
            button.setEnabled(False)

        form = QFormLayout()
        form.addRow("Media-language SRT", path_row(self.primary_path, self.primary_browse))
        form.addRow("Other-language SRT", path_row(self.secondary_path, self.secondary_browse))
        form.addRow("Output Prefix", self.output_prefix)
        form.addRow("Provider", self.provider)
        form.addRow("Model", self.model)
        form.addRow("Base URL", self.base_url)
        form.addRow("API Key", self.api_key)
        form.addRow("Timeout", self.timeout)
        form.addRow("Batch Size", self.batch_size)
        form.addRow("Media Lang", self.primary_language)
        form.addRow("Other Lang", self.secondary_language)

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.open_alignment_button)
        buttons.addWidget(self.load_button)

        review_header = QHBoxLayout()
        review_header.addWidget(QLabel("Human Review"))
        review_header.addStretch(1)
        review_header.addWidget(self.issues_only)

        review_buttons = QHBoxLayout()
        review_buttons.addWidget(self.approve_button)
        review_buttons.addWidget(self.apply_selection_button)
        review_buttons.addWidget(self.save_review_button)

        run_page = QWidget()
        run_layout = QVBoxLayout(run_page)
        run_layout.addLayout(form)
        run_layout.addWidget(self.export_srt)
        run_layout.addWidget(self.load_when_done)
        run_layout.addWidget(self.context_retry)
        run_layout.addWidget(self.repair_target_dialogue)
        run_layout.addLayout(buttons)
        run_layout.addWidget(QLabel("Progress"))
        run_layout.addWidget(self.log, 1)

        review_page = QWidget()
        review_layout = QVBoxLayout(review_page)
        review_layout.addLayout(review_header)
        review_layout.addWidget(self.review_summary)
        review_layout.addWidget(self.review_list, 1)
        review_layout.addWidget(QLabel("Media-language segment"))
        review_layout.addWidget(self.primary_review)
        review_layout.addWidget(QLabel("Candidate other-language cues"))
        review_layout.addWidget(self.candidate_list, 1)
        review_layout.addWidget(self.review_status)
        review_layout.addLayout(review_buttons)

        self.tabs = QTabWidget()
        self.tabs.addTab(run_page, "Align")
        self.tabs.addTab(review_page, "Review")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

        self.primary_browse.clicked.connect(lambda: self.browse_into(self.primary_path))
        self.secondary_browse.clicked.connect(lambda: self.browse_into(self.secondary_path))
        self.provider.currentTextChanged.connect(self.apply_provider_preset)
        self.context_retry.toggled.connect(self.update_repair_enabled)
        self.run_button.clicked.connect(self.emit_run)
        self.open_alignment_button.clicked.connect(self.open_alignment)
        self.load_button.clicked.connect(self.load_last_alignment)
        self.issues_only.toggled.connect(self.refresh_review_list)
        self.review_list.currentRowChanged.connect(self.show_review_segment)
        self.approve_button.clicked.connect(self.approve_current_segment)
        self.apply_selection_button.clicked.connect(self.apply_checked_cues)
        self.save_review_button.clicked.connect(self.save_alignment)
        self.setStyleSheet(
            """
            AlignmentPanel { color: #e5e7eb; background: #111827; }
            QLabel, QCheckBox { color: #e5e7eb; background: transparent; }
            QTabWidget::pane {
                background: #111827;
                border: 1px solid #334155;
            }
            QTabBar::tab {
                background: #1f2937;
                color: #cbd5e1;
                border: 1px solid #334155;
                padding: 6px 12px;
            }
            QTabBar::tab:selected {
                background: #334155;
                color: #f8fafc;
            }
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
                background: #0f172a;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background: #1f2937;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                min-height: 26px;
                padding: 0 10px;
            }
            """
        )

    def set_loaded_paths(
        self,
        primary: str | None,
        secondary: str | None,
        *,
        clear_missing: bool = False,
    ) -> None:
        if primary:
            self.primary_path.setText(primary)
        elif clear_missing:
            self.primary_path.clear()
        if secondary:
            self.secondary_path.setText(secondary)
        elif clear_missing:
            self.secondary_path.clear()

    def apply_llm_settings(self, settings: LLMEndpointSettings) -> None:
        provider = settings.provider if settings.provider in {"lmstudio", "openai", "deepseek", "custom"} else "custom"
        self.provider.blockSignals(True)
        self.provider.setCurrentText(provider)
        self.provider.blockSignals(False)
        self.model.setText(settings.model)
        self.base_url.setText(settings.base_url)
        self.api_key.setText(settings.api_key)
        self.timeout.setValue(int(settings.timeout))
        self.set_endpoint_fields_enabled(provider != "heuristic")

    def apply_provider_preset(self, provider: str) -> None:
        enabled = provider != "heuristic"
        self.set_endpoint_fields_enabled(enabled)
        if not enabled or provider == "custom":
            return
        preset = settings_from_provider(provider)
        self.model.setText(preset.model)
        self.base_url.setText(preset.base_url)
        self.api_key.setText(preset.api_key)
        self.timeout.setValue(int(preset.timeout))

    def set_endpoint_fields_enabled(self, enabled: bool) -> None:
        for field in (self.model, self.base_url, self.api_key, self.timeout):
            field.setEnabled(enabled)
        self.update_repair_enabled()

    def update_repair_enabled(self, _checked: bool | None = None) -> None:
        self.repair_target_dialogue.setEnabled(
            self.provider.currentText() != "heuristic"
            and self.context_retry.isChecked()
        )

    def emit_run(self) -> None:
        self.log.clear()
        self.append_log("Starting alignment...")
        self.run_button.setEnabled(False)
        self.run_requested.emit(self.config())

    def config(self) -> dict:
        return {
            "primary_path": self.primary_path.text(),
            "secondary_path": self.secondary_path.text(),
            "output_prefix": self.output_prefix.text(),
            "provider": self.provider.currentText(),
            "model": self.model.text(),
            "base_url": self.base_url.text(),
            "api_key": self.api_key.text(),
            "timeout": self.timeout.value(),
            "batch_size": self.batch_size.value(),
            "primary_language": self.primary_language.text(),
            "secondary_language": self.secondary_language.text(),
            "export_srt": self.export_srt.isChecked(),
            "context_retry": self.context_retry.isChecked(),
            "repair_target_dialogue": (
                self.repair_target_dialogue.isChecked()
                and self.repair_target_dialogue.isEnabled()
            ),
        }

    def mark_finished(self, result: dict) -> None:
        package = result["package"]
        sidecar_path = Path(result["sidecar_path"])
        exports = result.get("exports")
        self.run_button.setEnabled(True)
        self.set_review_package(package, sidecar_path)
        self.append_log(
            f"Finished. Segments: {len(package.segments)}. "
            f"Needs review: {len(package.issues)}."
        )
        self.append_log(f"Alignment: {sidecar_path}")
        if exports:
            self.append_log(f"Primary export: {exports[0]}")
            self.append_log(f"Secondary export: {exports[1]}")
        if self.load_when_done.isChecked():
            self.load_last_alignment()

    def mark_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.append_log(f"Error: {message}")

    def append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def set_review_package(
        self,
        package: AlignmentPackage,
        sidecar_path: str | Path,
    ) -> None:
        self.package = package
        self.sidecar_path = Path(sidecar_path)
        self.load_button.setEnabled(True)
        self.save_review_button.setEnabled(True)
        self.refresh_review_list()
        self.tabs.setCurrentIndex(1)

    def refresh_review_list(self, _checked: bool | None = None) -> None:
        selected_id = self.current_segment_id()
        self.review_list.clear()
        self._visible_segment_ids = []
        if self.package is None:
            self.review_summary.setText("Run or open an alignment to review it.")
            return

        unresolved = [
            segment
            for segment in self.package.segments
            if segment.status in {"needs_review", "needs_repair"}
        ]
        segment_ids = {
            primary_id
            for segment in self.package.segments
            for primary_id in (segment.primary_segment_ids or [segment.segment_id])
        }
        segment_ids.update(segment.segment_id for segment in self.package.segments)
        orphan_issues = [
            issue
            for issue in self.package.issues
            if issue.result.primary_id not in segment_ids
        ]
        self.review_summary.setText(
            f"{len(self.package.segments)} segments · {len(unresolved)} unresolved · "
            f"{len(orphan_issues)} extra response issues · "
            f"sidecar schema {self.package.schema_version}"
        )
        shown = unresolved if self.issues_only.isChecked() else self.package.segments
        for segment in shown:
            problems = f" · {'; '.join(segment.problems)}" if segment.problems else ""
            source_ids = segment.primary_segment_ids or [segment.segment_id]
            block_label = (
                source_ids[0]
                if len(source_ids) == 1
                else f"{source_ids[0]}–{source_ids[-1]}"
            )
            item = QListWidgetItem(
                f"{block_label} · {segment.status} · "
                f"{segment.start:.1f}s{problems}"
            )
            item.setData(Qt.ItemDataRole.UserRole, segment.segment_id)
            self.review_list.addItem(item)
            self._visible_segment_ids.append(segment.segment_id)

        if not shown:
            self.clear_review_detail("No unresolved segments remain.")
            return
        target_row = 0
        if selected_id in self._visible_segment_ids:
            target_row = self._visible_segment_ids.index(selected_id)
        self.review_list.setCurrentRow(target_row)

    def current_segment_id(self) -> str | None:
        item = self.review_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, str) else None

    def current_segment(self):
        segment_id = self.current_segment_id()
        if self.package is None or segment_id is None:
            return None
        return next(
            (segment for segment in self.package.segments if segment.segment_id == segment_id),
            None,
        )

    def show_review_segment(self, _row: int) -> None:
        segment = self.current_segment()
        if segment is None:
            self.clear_review_detail("")
            return
        self.primary_review.setPlainText(segment.primary_text)
        self.candidate_list.clear()
        selected = set(segment.secondary_cue_ids)
        for cue in segment.candidate_secondary_cues:
            item = QListWidgetItem(
                f"[{cue.cue_id}] {cue.start:.2f}–{cue.end:.2f}\n{cue.text}"
            )
            item.setData(Qt.ItemDataRole.UserRole, cue.cue_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if cue.cue_id in selected
                else Qt.CheckState.Unchecked
            )
            self.candidate_list.addItem(item)
        problem_text = "; ".join(segment.problems) or "No validator problems."
        model_note = (
            f"\nModel note: {segment.alignment_notes}"
            if segment.alignment_notes
            else ""
        )
        generated_note = ""
        if segment.generated_secondary is not None:
            generated = segment.generated_secondary
            generated_note = (
                f"\nGenerated {generated.target_language}: {generated.text}"
                f"\nProvenance: {generated.provider} / {generated.model} · "
                f"confidence {generated.confidence:.2f}"
            )
        self.review_status.setText(
            f"Status: {segment.status} · {segment.alignment_stage.replace('_', ' ')} · "
            f"{segment.disposition.replace('_', ' ')} · "
            f"confidence {segment.confidence:.2f} · {problem_text}"
            f"{model_note}{generated_note}"
        )
        self.review_status.setToolTip(segment.alignment_notes)
        self.approve_button.setEnabled(True)
        self.apply_selection_button.setEnabled(True)

    def clear_review_detail(self, message: str) -> None:
        self.primary_review.clear()
        self.candidate_list.clear()
        self.review_status.setText(message)
        self.approve_button.setEnabled(False)
        self.apply_selection_button.setEnabled(False)

    def approve_current_segment(self) -> None:
        segment = self.current_segment()
        if segment is None:
            return
        self.apply_review(None)

    def apply_checked_cues(self) -> None:
        selected_ids: list[str] = []
        for index in range(self.candidate_list.count()):
            item = self.candidate_list.item(index)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            cue_id = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(cue_id, str):
                selected_ids.append(cue_id)
        self.apply_review(selected_ids)

    def apply_review(self, selected_ids: list[str] | None) -> None:
        segment_id = self.current_segment_id()
        if self.package is None or segment_id is None:
            return
        self.package = review_alignment_segment(self.package, segment_id, selected_ids)
        self.save_alignment()
        self.refresh_review_list()

    def save_alignment(self) -> None:
        if self.package is None or self.sidecar_path is None:
            return
        write_alignment_sidecar(self.package, self.sidecar_path)
        self.append_log(f"Saved alignment: {self.sidecar_path}")

    def load_last_alignment(self) -> None:
        if self.sidecar_path is None:
            return
        self.load_alignment_requested.emit(str(self.sidecar_path))

    def open_alignment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open alignment sidecar",
            str(Path.home()),
            "Submerger alignment (*.alignment.json);;JSON files (*.json)",
        )
        if not path:
            return
        try:
            self.set_review_package(load_alignment_package(path), path)
            self.append_log(f"Opened alignment: {path}")
        except Exception as exc:  # noqa: BLE001 - present malformed sidecars in the panel.
            self.append_log(f"Error opening alignment: {exc}")

    def browse_into(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select subtitle", str(Path.home()), "SubRip subtitles (*.srt);;All files (*)")
        if path:
            target.setText(path)


def path_row(line_edit: QLineEdit, button: QPushButton) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line_edit, 1)
    layout.addWidget(button)
    return row
