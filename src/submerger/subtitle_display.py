from __future__ import annotations

from dataclasses import dataclass
from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QFrame, QTextEdit, QVBoxLayout, QWidget

from .interaction import SubtitleInteraction, SubtitleToken, tokenize_subtitle


@dataclass(frozen=True)
class SubtitleSnapshot:
    primary: str
    secondary: str
    timestamp: float | None


class InteractiveSubtitleLine(QTextEdit):
    token_hovered = Signal(object)
    token_clicked = Signal(object)
    phrase_selected = Signal(str, str)

    def __init__(self, *, language: str, color: str, font_size: int) -> None:
        super().__init__()
        self.language = language
        self.tokens: list[SubtitleToken] = []
        self._plain_text = ""
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        self.setStyleSheet(
            f"""
            QTextEdit {{
                background: transparent;
                border: 0;
                color: {color};
                font-size: {font_size}px;
                font-weight: 700;
                padding: 0 32px;
            }}
            """
        )
        self.document().setDocumentMargin(0)

    def set_subtitle_text(self, text: str) -> None:
        if text == self._plain_text:
            return
        self._plain_text = text
        self.tokens = tokenize_subtitle(text, self.language, text)
        self.setHtml(render_interactive_subtitle(text, self.language))
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
        self.setFixedHeight(max(42, int(self.document().size().height()) + 10))

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        token = self._token_at(event.position().toPoint())
        if token is not None:
            self.token_hovered.emit(token)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        selected = self.textCursor().selectedText().replace("\u2029", " ").strip()
        if selected:
            self.phrase_selected.emit(selected, self.language)
            cursor = self.textCursor()
            cursor.clearSelection()
            self.setTextCursor(cursor)
        else:
            token = self._token_at(event.position().toPoint())
            if token is not None:
                self.token_clicked.emit(token)
        super().mouseReleaseEvent(event)

    def _token_at(self, point) -> SubtitleToken | None:
        cursor = self.cursorForPosition(point)
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        word = cursor.selectedText().strip()
        if not word:
            return None
        normalized = word.lower().strip(".,!?;:\"'()[]{}“”‘’")
        for token in self.tokens:
            if token.normalized == normalized:
                return token
        return None


class SubtitleOverlay(QWidget):
    interaction_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = SubtitleSnapshot("", "", None)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        self.primary = InteractiveSubtitleLine(language="primary", color="#f8fafc", font_size=22)
        self.secondary = InteractiveSubtitleLine(language="secondary", color="#bfdbfe", font_size=18)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 92)
        layout.addStretch(1)
        layout.addWidget(self.primary)
        layout.addWidget(self.secondary)

        for line in (self.primary, self.secondary):
            line.token_hovered.connect(self._on_token_hovered)
            line.token_clicked.connect(self._on_token_clicked)
            line.phrase_selected.connect(self._on_phrase_selected)

    def set_subtitles(self, primary: str, secondary: str, timestamp: float | None = None) -> None:
        self.snapshot = SubtitleSnapshot(primary, secondary, timestamp)
        self.primary.set_subtitle_text(primary)
        self.secondary.set_subtitle_text(secondary)

    def _on_token_hovered(self, token: SubtitleToken) -> None:
        self.interaction_requested.emit(self._interaction("hover", token.text, token.language, token.index))

    def _on_token_clicked(self, token: SubtitleToken) -> None:
        self.interaction_requested.emit(self._interaction("click", token.text, token.language, token.index))

    def _on_phrase_selected(self, text: str, language: str) -> None:
        self.interaction_requested.emit(self._interaction("selection", text, language, None))

    def _interaction(self, kind: str, text: str, language: str, token_index: int | None) -> SubtitleInteraction:
        return SubtitleInteraction(
            kind=kind,
            text=text,
            language=language,
            timestamp=self.snapshot.timestamp,
            paired_text=self.snapshot.secondary if language == "primary" else self.snapshot.primary,
            token_index=token_index,
        )


def render_interactive_subtitle(text: str, language: str) -> str:
    if not text:
        return ""
    escaped = escape(text).replace("\n", "<br>")
    return (
        '<div align="center">'
        f'<span style="background-color: rgba(0, 0, 0, 150); padding: 2px 8px;" data-language="{language}">'
        f"{escaped}</span></div>"
    )
