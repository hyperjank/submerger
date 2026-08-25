from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from .settings import LLMEndpointSettings, settings_from_provider


class LLMSettingsDialog(QDialog):
    settings_saved = Signal(object)

    def __init__(self, settings: LLMEndpointSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LLM Endpoint Settings")
        self.provider = QComboBox()
        self.provider.addItems(["lmstudio", "openai", "deepseek", "custom"])
        self.model = QLineEdit()
        self.base_url = QLineEdit()
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.timeout = QSpinBox()
        self.timeout.setRange(10, 600)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(200, 8000)

        form = QFormLayout()
        form.addRow("Provider", self.provider)
        form.addRow("Model", self.model)
        form.addRow("Base URL", self.base_url)
        form.addRow("API Key", self.api_key)
        form.addRow("Timeout", self.timeout)
        form.addRow("Max Tokens", self.max_tokens)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.provider.currentTextChanged.connect(self.apply_provider_preset)
        self.set_settings(settings)
        self.setStyleSheet(
            """
            LLMSettingsDialog {
                background: #111827;
                color: #e5e7eb;
            }
            QLabel { color: #e5e7eb; }
            QLineEdit, QComboBox, QSpinBox {
                background: #0f172a;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 4px;
                min-height: 24px;
                padding: 3px 6px;
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

    def set_settings(self, settings: LLMEndpointSettings) -> None:
        self.provider.blockSignals(True)
        self.provider.setCurrentText(settings.provider)
        self.provider.blockSignals(False)
        self.model.setText(settings.model)
        self.base_url.setText(settings.base_url)
        self.api_key.setText(settings.api_key)
        self.timeout.setValue(int(settings.timeout))
        self.max_tokens.setValue(settings.max_tokens)

    def apply_provider_preset(self, provider: str) -> None:
        if provider == "custom":
            return
        preset = settings_from_provider(provider)
        self.model.setText(preset.model)
        self.base_url.setText(preset.base_url)
        self.api_key.setText(preset.api_key)
        self.timeout.setValue(int(preset.timeout))
        self.max_tokens.setValue(preset.max_tokens)

    def settings(self) -> LLMEndpointSettings:
        return LLMEndpointSettings(
            provider=self.provider.currentText(),
            model=self.model.text().strip(),
            base_url=self.base_url.text().strip(),
            api_key=self.api_key.text(),
            timeout=float(self.timeout.value()),
            max_tokens=self.max_tokens.value(),
        )

    def accept(self) -> None:  # type: ignore[override]
        self.settings_saved.emit(self.settings())
        super().accept()
