import json
import tempfile
import unittest
from pathlib import Path

from submerger.playback import (
    PlaybackSession,
    PlaybackSessionStore,
    classify_dropped_paths,
    discover_alignment_sidecar,
    discover_external_subtitles,
    is_text_subtitle_track,
    subtitle_track_label,
)


class PlaybackTests(unittest.TestCase):
    def test_discovers_language_tagged_subtitles_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            video = directory / "Episode.01.mkv"
            primary = directory / "Episode.01.en.srt"
            secondary = directory / "Episode.01.zh-Hans.srt"
            unrelated = directory / "Episode.02.en.srt"
            sidecar = directory / "Episode.01.alignment.json"
            for path in (video, primary, secondary, unrelated, sidecar):
                path.touch()

            found_primary, found_secondary = discover_external_subtitles(video)

            self.assertEqual(found_primary, primary)
            self.assertEqual(found_secondary, secondary)
            self.assertEqual(discover_alignment_sidecar(video), sidecar)

    def test_drop_classification_ignores_unknown_files(self) -> None:
        classified = classify_dropped_paths([
            "/tmp/episode.mkv",
            "/tmp/episode.en.srt",
            "/tmp/episode.alignment.json",
            "/tmp/readme.txt",
        ])

        self.assertEqual([path.name for path in classified["video"]], ["episode.mkv"])
        self.assertEqual([path.name for path in classified["subtitle"]], ["episode.en.srt"])
        self.assertEqual([path.name for path in classified["alignment"]], ["episode.alignment.json"])

    def test_session_store_orders_recents_and_round_trips_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = directory / "first.mkv"
            second = directory / "second.mkv"
            first.touch()
            second.touch()
            store = PlaybackSessionStore(directory / "state.json")
            store.remember(PlaybackSession(str(first), position=12.5))
            store.remember(PlaybackSession(
                str(second),
                position=9,
                speed=1.25,
                primary_offset=0.4,
                secondary_offset=-0.2,
                primary_embedded_id=3,
            ))

            sessions = store.sessions()

            self.assertEqual([item.title for item in sessions], ["second.mkv", "first.mkv"])
            self.assertEqual(sessions[0].speed, 1.25)
            self.assertEqual(sessions[0].primary_offset, 0.4)
            self.assertEqual(sessions[0].primary_embedded_id, 3)

    def test_session_store_ignores_malformed_or_missing_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({
                "version": 1,
                "sessions": [{"video_path": "/missing/movie.mkv"}, {"bad": True}],
            }))

            self.assertEqual(PlaybackSessionStore(path).sessions(), [])

    def test_embedded_track_label_uses_available_metadata(self) -> None:
        label = subtitle_track_label({
            "id": 4,
            "lang": "eng",
            "title": "English SDH",
            "codec": "subrip",
            "default": True,
        })

        self.assertEqual(label, "4: eng · English SDH · subrip · default")
        self.assertTrue(is_text_subtitle_track({"codec": "subrip"}))
        self.assertFalse(is_text_subtitle_track({"codec": "hdmv_pgs_subtitle"}))


if __name__ == "__main__":
    unittest.main()
