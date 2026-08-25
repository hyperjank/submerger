from __future__ import annotations

import html
import locale
import logging
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCursor, QOpenGLContext
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedLayout,
    QStyle,
    QTextBrowser,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
import mpv

from .alignment_panel import AlignmentPanel, AlignmentWorker
from .alignment import load_alignment_package, tracks_from_alignment_package
from .interaction import SubtitleInteraction
from .plugins import PluginContext, PluginRegistry, PluginResult, create_default_registry
from .script_sidebar import ScriptSidebar
from .settings import LLMEndpointSettings, load_llm_settings, save_llm_settings
from .settings_dialog import LLMSettingsDialog
from .subtitle_display import SubtitleOverlay
from .subtitles import DualSubtitleEngine


LOGGER = logging.getLogger(__name__)


class PluginWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class PluginWorker(QRunnable):
    def __init__(self, registry: PluginRegistry, plugin_id: str, action: str, context: PluginContext) -> None:
        super().__init__()
        self.registry = registry
        self.plugin_id = plugin_id
        self.action = action
        self.context = context
        self.signals = PluginWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.registry.run(self.plugin_id, self.action, self.context))
        except Exception as exc:  # noqa: BLE001 - surface plugin failures in the dock.
            self.signals.failed.emit(str(exc))


class DockTitleBar(QWidget):
    def __init__(self, title: str, dock: QDockWidget) -> None:
        super().__init__(dock)
        self.label = QLabel(title)
        self.float_button = QToolButton()
        self.float_button.setText("□")
        self.close_button = QToolButton()
        self.close_button.setText("×")
        self.float_button.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        self.close_button.clicked.connect(dock.hide)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.float_button)
        layout.addWidget(self.close_button)
        self.setStyleSheet(
            """
            DockTitleBar {
                background: #111827;
                color: #e5e7eb;
            }
            QLabel {
                color: #e5e7eb;
                font-weight: 600;
            }
            QToolButton {
                background: #1f2937;
                color: #e5e7eb;
                border: 1px solid #374151;
                border-radius: 3px;
                min-width: 18px;
                min-height: 18px;
            }
            """
        )


class MpvVideoWidget(QOpenGLWidget):
    render_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.player: mpv.MPV | None = None
        self.render_context: mpv.MpvRenderContext | None = None
        self._get_proc_address_callback = None
        self._needs_render = False
        self.render_requested.connect(self.update, Qt.ConnectionType.QueuedConnection)
        self.setStyleSheet("background: #050505;")

    def initializeGL(self) -> None:
        pass

    def resizeGL(self, _width: int, _height: int) -> None:
        self.update()

    def paintGL(self) -> None:
        if self.render_context is None:
            return

        ratio = self.devicePixelRatioF()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))
        self.render_context.update()
        self._needs_render = False
        self.render_context.render(
            opengl_fbo={
                "w": width,
                "h": height,
                "fbo": self.defaultFramebufferObject(),
            },
            flip_y=True,
        )
        self.render_context.report_swap()

    def _ensure_player(self) -> mpv.MPV:
        if self.player is not None:
            return self.player

        locale.setlocale(locale.LC_NUMERIC, "C")
        self.player = mpv.MPV(
            vo="libmpv",
            input_default_bindings=True,
            input_vo_keyboard=True,
            osc=False,
            sub_auto="no",
            sid="no",
            keep_open=True,
        )
        self._get_proc_address_callback = mpv.MpvGlGetProcAddressFn(self._get_proc_address)
        self.render_context = mpv.MpvRenderContext(
            self.player,
            "opengl",
            opengl_init_params={"get_proc_address": self._get_proc_address_callback},
        )
        self.render_context.update_cb = self._on_mpv_render_update
        return self.player

    def _get_proc_address(self, _ctx, name: bytes) -> int:
        context = QOpenGLContext.currentContext()
        if context is None:
            LOGGER.debug("No current OpenGL context while resolving %r", name)
            return 0

        try:
            address = context.getProcAddress(name)
        except TypeError:
            address = context.getProcAddress(name.decode("ascii"))
        return int(address or 0)

    def _on_mpv_render_update(self) -> None:
        self._needs_render = True
        self.render_requested.emit()

    def shutdown(self) -> None:
        self.makeCurrent()
        try:
            if self.render_context is not None:
                self.render_context.update_cb = None
                self.render_context.free()
                self.render_context = None
                self._get_proc_address_callback = None
            if self.player is not None:
                self.player.terminate()
                self.player = None
        finally:
            self.doneCurrent()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Submerger")
        self.resize(1100, 720)

        self.subtitle_engine = DualSubtitleEngine()
        self.llm_settings = load_llm_settings()
        self.plugin_registry = create_default_registry(llm_settings=self.llm_settings)
        self.thread_pool = QThreadPool.globalInstance()
        self.duration = 0.0
        self._seeking = False
        self._last_hover_text = ""
        self.current_plugin_context: PluginContext | None = None
        self.plugin_action_buttons: list[QPushButton] = []
        self.primary_subtitle_path: str | None = None
        self.secondary_subtitle_path: str | None = None
        self.alignment_sidecar_path: str | None = None

        self.video_surface = MpvVideoWidget()
        self.overlay = SubtitleOverlay()

        self.open_video_button = QPushButton("Open Video")
        self.open_primary_button = QPushButton("Primary SRT")
        self.open_secondary_button = QPushButton("Secondary SRT")
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_label = QLabel("00:00 / 00:00")
        self.plugin_output = QTextBrowser()
        self.plugin_actions_bar = QHBoxLayout()
        self.plugin_panel = QWidget()
        self.plugin_dock = QDockWidget("Language Tools", self)
        self.script_button = QPushButton("Script")
        self.script_sidebar = ScriptSidebar()
        self.script_dock = QDockWidget("Episode Script", self)
        self.align_button = QPushButton("Align")
        self.alignment_panel = AlignmentPanel()
        self.alignment_panel.apply_llm_settings(self.llm_settings)
        self.alignment_dock = QDockWidget("Subtitle Alignment", self)

        self._build_layout()
        self._build_menu()
        self._connect()

        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def _ensure_player(self) -> mpv.MPV:
        self.video_surface.makeCurrent()
        try:
            return self.video_surface._ensure_player()
        finally:
            self.video_surface.doneCurrent()

    def _build_layout(self) -> None:
        stage = QFrame()
        stage.setFrameShape(QFrame.Shape.NoFrame)
        stack = QStackedLayout(stage)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        stack.addWidget(self.video_surface)
        stack.addWidget(self.overlay)
        stack.setCurrentWidget(self.overlay)
        self.overlay.raise_()

        controls = QHBoxLayout()
        controls.setContentsMargins(10, 8, 10, 8)
        controls.setSpacing(8)
        controls.addWidget(self.open_video_button)
        controls.addWidget(self.open_primary_button)
        controls.addWidget(self.open_secondary_button)
        controls.addWidget(self.script_button)
        controls.addWidget(self.align_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.position_slider, 1)
        controls.addWidget(self.time_label)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(stage, 1)
        root.addLayout(controls)

        container = QWidget()
        container.setObjectName("root")
        container.setLayout(root)
        container.setStyleSheet(
            """
            QWidget#root { background: #111827; color: #e5e7eb; }
            QPushButton {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                color: #f9fafb;
                min-height: 30px;
                padding: 0 12px;
            }
            QPushButton:hover { background: #273244; }
            QLabel { color: #d1d5db; }
            SubtitleOverlay { background: transparent; }
            """
        )
        self.setCentralWidget(container)

        self.plugin_output.setReadOnly(True)
        self.plugin_output.setPlainText("Hover a word for dictionary context. Drag-select a phrase for explanation.")
        self.plugin_output.setStyleSheet(
            """
            QTextBrowser {
                background: #0f172a;
                color: #e2e8f0;
                border: 0;
                font-size: 13px;
                padding: 10px;
            }
            """
        )
        self.plugin_actions_bar.setContentsMargins(8, 8, 8, 0)
        self.plugin_actions_bar.addStretch(1)
        plugin_layout = QVBoxLayout(self.plugin_panel)
        plugin_layout.setContentsMargins(0, 0, 0, 0)
        plugin_layout.setSpacing(0)
        plugin_layout.addLayout(self.plugin_actions_bar)
        plugin_layout.addWidget(self.plugin_output, 1)
        self.plugin_dock.setWidget(self.plugin_panel)
        self.plugin_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.plugin_dock.setTitleBarWidget(DockTitleBar("Language Tools", self.plugin_dock))
        self.plugin_dock.setMinimumWidth(300)
        self.plugin_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.plugin_dock)

        self.script_dock.setWidget(self.script_sidebar)
        self.script_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.script_dock.setTitleBarWidget(DockTitleBar("Episode Script", self.script_dock))
        self.script_dock.setMinimumWidth(300)
        self.script_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.script_dock)

        self.alignment_dock.setWidget(self.alignment_panel)
        self.alignment_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.alignment_dock.setTitleBarWidget(DockTitleBar("Subtitle Alignment", self.alignment_dock))
        self.alignment_dock.setMinimumWidth(360)
        self.alignment_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.alignment_dock)

    def _build_menu(self) -> None:
        settings_menu = self.menuBar().addMenu("Settings")
        llm_action = QAction("LLM Endpoint Settings", self)
        llm_action.triggered.connect(self.open_llm_settings)
        settings_menu.addAction(llm_action)

    def _connect(self) -> None:
        self.open_video_button.clicked.connect(self.open_video)
        self.open_primary_button.clicked.connect(lambda: self.open_subtitle(primary=True))
        self.open_secondary_button.clicked.connect(lambda: self.open_subtitle(primary=False))
        self.play_button.clicked.connect(self.toggle_playback)
        self.script_button.clicked.connect(self.toggle_script_sidebar)
        self.align_button.clicked.connect(self.toggle_alignment_panel)
        self.script_sidebar.seek_requested.connect(self.seek_to)
        self.alignment_panel.run_requested.connect(self.run_alignment)
        self.alignment_panel.load_alignment_requested.connect(self.load_alignment_sidecar)
        self.position_slider.sliderPressed.connect(self._begin_seek)
        self.position_slider.sliderReleased.connect(self._end_seek)
        self.overlay.interaction_requested.connect(self.handle_subtitle_interaction)

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            str(Path.home()),
            "Video files (*.mkv *.mp4 *.avi *.mov *.webm);;All files (*)",
        )
        if path:
            player = self._ensure_player()
            player.command("loadfile", path)
            player.pause = False

    def open_subtitle(self, *, primary: bool) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open subtitle",
            str(Path.home()),
            "SubRip subtitles (*.srt);;All files (*)",
        )
        if not path:
            return

        try:
            if primary:
                self.subtitle_engine.load_primary(path)
                self.primary_subtitle_path = path
            else:
                self.subtitle_engine.load_secondary(path)
                self.secondary_subtitle_path = path
            self.script_sidebar.set_tracks(self.subtitle_engine.primary, self.subtitle_engine.secondary)
            self.alignment_panel.set_loaded_paths(self.primary_subtitle_path, self.secondary_subtitle_path)
        except Exception as exc:  # noqa: BLE001 - show parser/file failures to the user.
            QMessageBox.critical(self, "Subtitle error", str(exc))

    def toggle_playback(self) -> None:
        player = self._ensure_player()
        player.pause = not bool(player.pause)

    def refresh(self) -> None:
        player = self.video_surface.player
        if player is None:
            self.overlay.set_subtitles("", "", None)
            return

        timestamp = player.time_pos
        self.duration = float(player.duration or self.duration or 0.0)

        primary, secondary = self.subtitle_engine.active(timestamp)
        self.overlay.set_subtitles(primary, secondary, timestamp)
        self.script_sidebar.update_position(timestamp)

        if not self._seeking and self.duration > 0 and timestamp is not None:
            self.position_slider.setRange(0, int(self.duration * 1000))
            self.position_slider.setValue(int(float(timestamp) * 1000))

        self.time_label.setText(f"{format_time(timestamp)} / {format_time(self.duration)}")
        icon = QStyle.StandardPixmap.SP_MediaPlay if player.pause else QStyle.StandardPixmap.SP_MediaPause
        self.play_button.setIcon(self.style().standardIcon(icon))

    def _begin_seek(self) -> None:
        self._seeking = True

    def _end_seek(self) -> None:
        self._seeking = False
        player = self.video_surface.player
        if player is not None:
            self.seek_to(self.position_slider.value() / 1000.0)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.video_surface.shutdown()
        super().closeEvent(event)

    def handle_subtitle_interaction(self, interaction) -> None:
        if interaction.kind == "hover" and interaction.text == self._last_hover_text:
            return
        if interaction.kind == "hover":
            self._last_hover_text = interaction.text

        if interaction.kind == "selection":
            self.show_plugin_actions(interaction)
            return

        context = self.plugin_context_for(interaction)
        actions = self.plugin_registry.actions_for_event(interaction.kind)
        if not actions:
            return
        action = actions[0]
        result = self.plugin_registry.run(action.plugin_id, action.label, context)
        if interaction.kind == "hover":
            QToolTip.showText(QCursor.pos(), tooltip_text(result), self.overlay)
            return

        self.show_plugin_result(result)

    def show_plugin_actions(self, interaction: SubtitleInteraction) -> None:
        context = self.plugin_context_for(interaction)
        self.current_plugin_context = context
        lines = [
            f"Selected segment at {format_time(interaction.timestamp)}",
            "",
            f"Primary:\n{context.primary_text or '(none)'}",
            "",
            f"Secondary:\n{context.secondary_text or '(none)'}",
            "",
            "Available plugin actions:",
        ]
        for action in self.plugin_registry.actions_for_event("selection"):
            lines.append(f"- {action.label}")
        self.show_plugin_result(PluginResult("Language Tools", "\n".join(lines), "plugin-registry"))
        self.set_plugin_actions(context, "selection")

    def set_plugin_actions(self, context: PluginContext, event_kind: str) -> None:
        while self.plugin_actions_bar.count() > 1:
            item = self.plugin_actions_bar.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.plugin_action_buttons = []

        for action in self.plugin_registry.actions_for_event(event_kind):
            button = QPushButton(action.label)
            button.clicked.connect(
                lambda _checked=False, plugin_id=action.plugin_id, label=action.label: self.run_plugin_action(
                    plugin_id,
                    label,
                    context,
                    async_run=plugin_id in {"phrase_explanation", "sentence_diagram"},
                )
            )
            self.plugin_actions_bar.insertWidget(max(0, self.plugin_actions_bar.count() - 1), button)
            self.plugin_action_buttons.append(button)

    def run_plugin_action(self, plugin_id: str, action: str, context: PluginContext, *, async_run: bool) -> None:
        if not async_run:
            self.show_plugin_result(self.plugin_registry.run(plugin_id, action, context))
            return

        self.show_plugin_result(PluginResult(f"Running {action}", "Working...", f"plugin:{plugin_id}:pending"))
        worker = PluginWorker(self.plugin_registry, plugin_id, action, context)
        worker.signals.finished.connect(self.show_plugin_result)
        worker.signals.failed.connect(self.show_plugin_error)
        self.thread_pool.start(worker)

    def plugin_context_for(self, interaction: SubtitleInteraction) -> PluginContext:
        primary = self.overlay.snapshot.primary
        secondary = self.overlay.snapshot.secondary
        return PluginContext(
            interaction=interaction,
            primary_text=primary,
            secondary_text=secondary,
            timestamp=interaction.timestamp,
        )

    def show_plugin_result(self, result: PluginResult) -> None:
        self.plugin_dock.show()
        self.plugin_dock.setWindowTitle(result.title)
        if result.content_type == "html":
            self.plugin_output.setHtml(f"{result.body}<p style='color:#94a3b8;'>Source: {html_escape(result.source)}</p>")
        else:
            self.plugin_output.setPlainText(f"{result.title}\n\n{result.body}\n\nSource: {result.source}")

    def show_plugin_error(self, message: str) -> None:
        self.show_plugin_result(PluginResult("Language Tool Error", message, "plugin:error"))

    def open_llm_settings(self) -> None:
        dialog = LLMSettingsDialog(self.llm_settings, self)
        dialog.settings_saved.connect(self.apply_llm_settings)
        dialog.exec()

    def apply_llm_settings(self, settings: LLMEndpointSettings) -> None:
        self.llm_settings = settings
        save_llm_settings(settings)
        self.plugin_registry = create_default_registry(llm_settings=settings)
        self.alignment_panel.apply_llm_settings(settings)

    def toggle_script_sidebar(self) -> None:
        self.script_dock.setVisible(not self.script_dock.isVisible())

    def toggle_alignment_panel(self) -> None:
        self.alignment_panel.set_loaded_paths(self.primary_subtitle_path, self.secondary_subtitle_path)
        self.alignment_dock.setVisible(not self.alignment_dock.isVisible())

    def run_alignment(self, config: dict) -> None:
        worker = AlignmentWorker(config)
        worker.signals.progress.connect(self.alignment_panel.append_log)
        worker.signals.finished.connect(self.alignment_panel.mark_finished)
        worker.signals.failed.connect(self.alignment_panel.mark_failed)
        self.thread_pool.start(worker)

    def load_alignment_sidecar(self, sidecar_path: str) -> None:
        try:
            package = load_alignment_package(sidecar_path)
            primary, secondary = tracks_from_alignment_package(package)
            self.subtitle_engine.primary = primary
            self.subtitle_engine.secondary = secondary
            self.primary_subtitle_path = package.primary_source
            self.secondary_subtitle_path = package.secondary_source
            self.alignment_sidecar_path = sidecar_path
            self.script_sidebar.set_tracks(self.subtitle_engine.primary, self.subtitle_engine.secondary)
            self.alignment_panel.set_loaded_paths(
                self.primary_subtitle_path,
                self.secondary_subtitle_path,
            )
        except Exception as exc:  # noqa: BLE001 - show parser/file failures to the user.
            QMessageBox.critical(self, "Alignment load error", str(exc))

    def seek_to(self, timestamp: float) -> None:
        player = self.video_surface.player
        if player is not None:
            player.seek(timestamp, reference="absolute")


def format_time(value: float | None) -> str:
    if value is None:
        value = 0
    total_seconds = max(0, int(value))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def tooltip_text(result: PluginResult) -> str:
    body = result.body.strip()
    if len(body) > 220:
        body = body[:217].rstrip() + "..."
    return f"{result.title}\n\n{body}"


def html_escape(value: str) -> str:
    return html.escape(value, quote=True)
