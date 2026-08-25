import unittest

from submerger.main import parse_cli_arguments


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


if __name__ == "__main__":
    unittest.main()
