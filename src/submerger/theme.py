from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


APPLICATION_STYLESHEET = """
QMainWindow, QDialog, QDockWidget {
    background: #111827;
    color: #e5e7eb;
}
QMenuBar {
    background: #111827;
    color: #e5e7eb;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 8px;
}
QMenuBar::item:selected, QMenuBar::item:pressed {
    background: #334155;
}
QMenu {
    background: #111827;
    color: #e5e7eb;
    border: 1px solid #475569;
}
QMenu::item:selected {
    background: #2563eb;
    color: #f8fafc;
}
QToolTip {
    background-color: #111827;
    color: #f8fafc;
    border: 1px solid #64748b;
    padding: 5px;
}
"""


def apply_application_theme(app: QApplication) -> None:
    """Apply one palette to native Qt surfaces and app-owned widgets."""
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#111827",
        QPalette.ColorRole.WindowText: "#e5e7eb",
        QPalette.ColorRole.Base: "#0f172a",
        QPalette.ColorRole.AlternateBase: "#172033",
        QPalette.ColorRole.ToolTipBase: "#111827",
        QPalette.ColorRole.ToolTipText: "#f8fafc",
        QPalette.ColorRole.Text: "#e2e8f0",
        QPalette.ColorRole.Button: "#1f2937",
        QPalette.ColorRole.ButtonText: "#f8fafc",
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Link: "#7dd3fc",
        QPalette.ColorRole.Highlight: "#2563eb",
        QPalette.ColorRole.HighlightedText: "#f8fafc",
        QPalette.ColorRole.PlaceholderText: "#94a3b8",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor("#64748b"),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor("#64748b"),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor("#64748b"),
    )
    app.setPalette(palette)
    app.setStyleSheet(APPLICATION_STYLESHEET)
