import unittest

from submerger.main import parse_cli_arguments

try:
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication
    from submerger.theme import APPLICATION_STYLESHEET, apply_application_theme
except ModuleNotFoundError:
    QApplication = None


class MainTests(unittest.TestCase):
    def test_cli_accepts_episode_and_subtitle_paths(self) -> None:
        args = parse_cli_arguments([
            "episode.mkv",
            "--primary",
            "episode.en.srt",
            "--secondary",
            "episode.zh.srt",
            "--no-restore",
        ])

        self.assertEqual(args.video, "episode.mkv")
        self.assertEqual(args.primary, "episode.en.srt")
        self.assertEqual(args.secondary, "episode.zh.srt")
        self.assertTrue(args.no_restore)

    @unittest.skipIf(QApplication is None, "PySide6 is not installed for this interpreter")
    def test_application_theme_sets_dark_surfaces_and_readable_tooltips(self) -> None:
        app = QApplication.instance() or QApplication([])

        apply_application_theme(app)

        palette = app.palette()
        self.assertEqual(palette.color(QPalette.ColorRole.Window).name(), "#111827")
        self.assertEqual(palette.color(QPalette.ColorRole.WindowText).name(), "#e5e7eb")
        self.assertEqual(palette.color(QPalette.ColorRole.ToolTipBase).name(), "#111827")
        self.assertEqual(palette.color(QPalette.ColorRole.ToolTipText).name(), "#f8fafc")
        self.assertIn("QToolTip", APPLICATION_STYLESHEET)
        self.assertIn("color: #f8fafc", APPLICATION_STYLESHEET)


if __name__ == "__main__":
    unittest.main()
