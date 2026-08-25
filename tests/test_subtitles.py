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

    def test_dual_engine_applies_independent_positive_delays(self) -> None:
        engine = DualSubtitleEngine()
        engine.primary = SubtitleTrack(parse_srt("1\n00:00:01,000 --> 00:00:02,000\nHello\n"))
        engine.secondary = SubtitleTrack(parse_srt("1\n00:00:01,000 --> 00:00:02,000\nHola\n"))

        self.assertEqual(engine.active(2.25, 1.0, 0.5), ("Hello", "Hola"))
        self.assertEqual(engine.active(1.25, 1.0, 0.5), ("", ""))

    def test_track_finds_current_previous_and_next_cues(self) -> None:
        track = SubtitleTrack(parse_srt(SRT))

        self.assertEqual(track.cue_at_or_before(3.75).text, "Hello & welcome")
        self.assertEqual(track.adjacent_cue(1.5, 1).text, "Line one\nLine two")
        self.assertIsNone(track.adjacent_cue(1.5, -1))
        self.assertEqual(track.adjacent_cue(5, -1).text, "Hello & welcome")


if __name__ == "__main__":
    unittest.main()
