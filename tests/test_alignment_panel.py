import unittest
import tempfile
from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from submerger.alignment import AlignmentPackage, PairedSegment
    from submerger.alignment_panel import AlignmentPanel
    from submerger.settings import LLMEndpointSettings
    from submerger.subtitles import SubtitleCue
except ModuleNotFoundError:
    QApplication = None
    AlignmentPanel = None
    LLMEndpointSettings = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed for this interpreter")
class AlignmentPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_config_includes_lmstudio_fields(self) -> None:
        panel = AlignmentPanel()
        panel.primary_path.setText("primary.srt")
        panel.secondary_path.setText("secondary.srt")
        panel.output_prefix.setText("episode-gui")

        config = panel.config()

        self.assertEqual(config["primary_path"], "primary.srt")
        self.assertEqual(config["secondary_path"], "secondary.srt")
        self.assertEqual(config["base_url"], "http://192.168.86.113:1234/v1")
        self.assertEqual(config["model"], "qwen3.5-4b")
        self.assertEqual(config["provider"], "lmstudio")

    def test_apply_llm_settings_updates_endpoint_fields(self) -> None:
        panel = AlignmentPanel()
        panel.apply_llm_settings(
            LLMEndpointSettings(
                provider="deepseek",
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                api_key="secret",
                timeout=90,
            )
        )

        config = panel.config()

        self.assertEqual(config["provider"], "deepseek")
        self.assertEqual(config["model"], "deepseek-chat")
        self.assertEqual(config["base_url"], "https://api.deepseek.com/v1")
        self.assertEqual(config["timeout"], 90)

    def test_changing_provider_applies_its_complete_endpoint_preset(self) -> None:
        panel = AlignmentPanel()
        panel.model.setText("stale-model")
        panel.base_url.setText("http://stale.invalid/v1")
        panel.api_key.setText("stale-key")
        panel.timeout.setValue(42)

        panel.provider.setCurrentText("deepseek")

        config = panel.config()
        self.assertEqual(config["model"], "deepseek-chat")
        self.assertEqual(config["base_url"], "https://api.deepseek.com/v1")
        self.assertNotEqual(config["api_key"], "stale-key")
        self.assertEqual(config["timeout"], 120)

    def test_heuristic_disables_endpoint_fields_without_destroying_them(self) -> None:
        panel = AlignmentPanel()
        old_endpoint = panel.base_url.text()

        panel.provider.setCurrentText("heuristic")

        self.assertFalse(panel.model.isEnabled())
        self.assertFalse(panel.base_url.isEnabled())
        self.assertFalse(panel.api_key.isEnabled())
        self.assertFalse(panel.timeout.isEnabled())
        self.assertEqual(panel.base_url.text(), old_endpoint)

        panel.provider.setCurrentText("lmstudio")
        self.assertTrue(panel.base_url.isEnabled())
        self.assertEqual(panel.base_url.text(), "http://192.168.86.113:1234/v1")

    def test_review_panel_applies_checked_source_cues_and_saves_sidecar(self) -> None:
        cue = SubtitleCue(1, 2, "Hola.", "s1", "Hola.")
        package = AlignmentPackage(
            primary_language="en",
            secondary_language="es",
            segments=[
                PairedSegment(
                    "p_1",
                    1,
                    2,
                    "Hello.",
                    "",
                    ["p1"],
                    [],
                    0,
                    "needs_review",
                    ["missing alignment result"],
                    [cue],
                )
            ],
            issues=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "episode.alignment.json"
            panel = AlignmentPanel()
            panel.set_review_package(package, sidecar)

            self.assertEqual(panel.review_list.count(), 1)
            self.assertEqual(panel.candidate_list.count(), 1)
            panel.candidate_list.item(0).setCheckState(Qt.CheckState.Checked)
            panel.apply_checked_cues()

            self.assertTrue(sidecar.exists())
            self.assertEqual(panel.package.segments[0].status, "reviewed")
            self.assertEqual(panel.package.segments[0].secondary_text, "Hola.")
            self.assertEqual(panel.review_list.count(), 0)


if __name__ == "__main__":
    unittest.main()
