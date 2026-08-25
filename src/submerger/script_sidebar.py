from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from .script_model import ScriptRow, build_script_rows, render_script_row
from .subtitles import SubtitleTrack


class ScriptSidebar(QWidget):
    seek_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[ScriptRow] = []
        self.active_index: int | None = None

        self.highlight_toggle = QCheckBox("Highlight")
        self.highlight_toggle.setChecked(True)
        self.autoscroll_toggle = QCheckBox("Auto-scroll")
        self.autoscroll_toggle.setChecked(True)
        self.empty_label = QLabel("Load subtitles to show the episode script.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.list_widget = QListWidget()
        self.list_widget.setWordWrap(True)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        controls = QHBoxLayout()
        controls.setContentsMargins(8, 8, 8, 4)
        controls.addWidget(self.highlight_toggle)
        controls.addWidget(self.autoscroll_toggle)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.list_widget, 1)

        self.list_widget.hide()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.highlight_toggle.toggled.connect(lambda _checked: self.update_position(None))
        self.setStyleSheet(
            """
            ScriptSidebar {
                background: #111827;
                color: #e5e7eb;
            }
            QListWidget {
                background: #0f172a;
                color: #dbeafe;
                border: 0;
                padding: 6px;
            }
            QListWidget::item {
                border-bottom: 1px solid #1e293b;
                padding: 8px;
            }
            QListWidget::item:selected {
                background: #1e3a5f;
                color: #f8fafc;
            }
            QCheckBox, QLabel {
                color: #e5e7eb;
            }
            """
        )

    def set_tracks(self, primary: SubtitleTrack, secondary: SubtitleTrack) -> None:
        self.rows = build_script_rows(primary, secondary)
        self.active_index = None
        self.list_widget.clear()
        for index, row in enumerate(self.rows):
            item = QListWidgetItem(render_script_row(row))
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setSizeHint(QSize(220, script_row_height(item.text())))
            self.list_widget.addItem(item)

        has_rows = bool(self.rows)
        self.empty_label.setVisible(not has_rows)
        self.list_widget.setVisible(has_rows)

    def update_position(self, timestamp: float | None) -> None:
        if timestamp is None or not self.rows:
            self._set_active_index(None)
            return

        active = next((index for index, row in enumerate(self.rows) if row.contains(timestamp)), None)
        self._set_active_index(active)

    def _set_active_index(self, index: int | None) -> None:
        if not self.highlight_toggle.isChecked():
            self.list_widget.clearSelection()
            self.active_index = None
            return
        if index == self.active_index:
            return

        self.active_index = index
        self.list_widget.clearSelection()
        if index is None:
            return

        item = self.list_widget.item(index)
        if item is None:
            return
        item.setSelected(True)
        self.list_widget.setCurrentItem(item)
        if self.autoscroll_toggle.isChecked():
            self.list_widget.scrollToItem(item, QListWidget.ScrollHint.PositionAtCenter)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(index, int) and 0 <= index < len(self.rows):
            self.seek_requested.emit(self.rows[index].start)


def script_row_height(text: str) -> int:
    lines = text.count("\n") + 1
    chars = max((len(line) for line in text.splitlines()), default=0)
    wrapped_lines = max(lines, lines + chars // 42)
    return min(180, max(78, wrapped_lines * 20 + 18))
