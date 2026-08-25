import unittest

from submerger.subtitles import DualSubtitleEngine, SubtitleTrack, parse_srt, parse_timestamp


SRT = """1
00:00:01,000 --> 00:00:03,500
<i>Hello</i> &amp; welcome

2
00:00:04.000 --> 00:00:06.000
Line one
Line two
"""


class SubtitleTests(unittest.TestCase):
    def test_parse_timestamp_accepts_comma_and_dot(self) -> None:
        self.assertEqual(parse_timestamp("01:02:03,500"), 3723.5)
        self.assertEqual(parse_timestamp("00:00:03.250"), 3.25)

    def test_parse_srt_strips_common_tags_and_preserves_lines(self) -> None:
        cues = parse_srt(SRT)

        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "Hello & welcome")
        self.assertEqual(cues[1].text, "Line one\nLine two")

    def test_subtitle_track_returns_active_text(self) -> None:
        track = SubtitleTrack(parse_srt(SRT))

        self.assertEqual(track.active_text(0.5), "")
        self.assertEqual(track.active_text(1.2), "Hello & welcome")
        self.assertEqual(track.active_text(3.5), "")
        self.assertEqual(track.active_text(4.2), "Line one\nLine two")

    def test_dual_engine_combines_tracks(self) -> None:
        engine = DualSubtitleEngine()
        engine.primary = SubtitleTrack(parse_srt(SRT))
        engine.secondary = SubtitleTrack(parse_srt("1\n00:00:01,000 --> 00:00:02,000\nHola\n"))

        self.assertEqual(engine.active(1.5), ("Hello & welcome", "Hola"))


if __name__ == "__main__":
    unittest.main()
