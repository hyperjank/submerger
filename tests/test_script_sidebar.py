import unittest

from submerger.script_model import build_script_rows, render_script_row
from submerger.subtitles import SubtitleCue, SubtitleTrack


class ScriptSidebarTests(unittest.TestCase):
    def test_build_script_rows_uses_primary_timing_and_secondary_midpoint(self) -> None:
        primary = SubtitleTrack([
            SubtitleCue(1, 3, "Hello.", "1"),
            SubtitleCue(4, 6, "Goodbye.", "2"),
        ])
        secondary = SubtitleTrack([
            SubtitleCue(1.5, 2.5, "Hola.", "1"),
            SubtitleCue(4.5, 5.5, "Adiós.", "2"),
        ])

        rows = build_script_rows(primary, secondary)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].primary, "Hello.")
        self.assertEqual(rows[0].secondary, "Hola.")
        self.assertTrue(rows[1].contains(5))

    def test_build_script_rows_falls_back_to_secondary_only(self) -> None:
        rows = build_script_rows(SubtitleTrack(), SubtitleTrack([SubtitleCue(1, 2, "Hola.", "1")]))

        self.assertEqual(rows[0].primary, "")
        self.assertEqual(rows[0].secondary, "Hola.")

    def test_render_script_row_includes_timestamp_and_languages(self) -> None:
        text = render_script_row(build_script_rows(
            SubtitleTrack([SubtitleCue(1, 2, "Hello.", "1")]),
            SubtitleTrack([SubtitleCue(1, 2, "Hola.", "1")]),
        )[0])

        self.assertIn("00:00:01,000", text)
        self.assertIn("Hello.", text)
        self.assertIn("Hola.", text)


if __name__ == "__main__":
    unittest.main()
