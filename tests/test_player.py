import tempfile
import unittest
from pathlib import Path

try:
    from PySide6.QtWidgets import QApplication
    from submerger.interaction import SubtitleInteraction
    from submerger.playback import PlaybackSessionStore
    from submerger.player import MainWindow
    from submerger.subtitles import SubtitleCue, SubtitleTrack
except ModuleNotFoundError:
    QApplication = None
    MainWindow = None


class FakePlayer:
    def __init__(self) -> None:
        self.time_pos = 5.0
        self.duration = 60.0
        self.pause = False
        self.speed = 1.0
        self.sub_delay = 0.0
        self.secondary_sub_delay = 0.0
        self.sid = "no"
        self.secondary_sid = "no"
        self.sub_text = "Embedded primary"
        self.secondary_sub_text = "Embedded secondary"
        self.track_list = []
        self.seeks: list[tuple] = []
        self.commands: list[tuple] = []

    def seek(self, *args, **kwargs) -> None:
        self.seeks.append((*args, kwargs))

    def command(self, *args) -> None:
        self.commands.append(args)

    def terminate(self) -> None:
        pass


@unittest.skipIf(QApplication is None, "PySide6 is not installed for this interpreter")
class PlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, directory: Path) -> tuple[MainWindow, FakePlayer]:
        window = MainWindow(
            session_store=PlaybackSessionStore(directory / "state.json"),
            restore_session=False,
        )
        window.timer.stop()
        player = FakePlayer()
        window.video_surface.player = player
        return window, player

    def test_replay_and_navigation_use_primary_cue_timing_and_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window, player = self.make_window(Path(tmp))
            window.subtitle_engine.primary = SubtitleTrack([
                SubtitleCue(1, 3, "One"),
                SubtitleCue(4, 6, "Two"),
                SubtitleCue(8, 10, "Three"),
            ])
            window.primary_offset = 0.5
            player.time_pos = 5.0

            window.replay_current_line()
            window.navigate_subtitle(1)
            window.navigate_subtitle(-1)

            self.assertEqual(player.seeks[0][0], 4.5)
            self.assertEqual(player.seeks[1][0], 8.5)
            self.assertEqual(player.seeks[2][0], 1.5)
            window.close()

    def test_embedded_tracks_autoselect_by_language_and_feed_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            video = directory / "episode.mkv"
            video.touch()
            window, player = self.make_window(directory)
            window.current_video_path = str(video)
            player.track_list = [
                {"id": 2, "type": "sub", "lang": "zho", "title": "Chinese", "codec": "subrip"},
                {"id": 3, "type": "sub", "lang": "eng", "title": "English", "codec": "subrip", "default": True},
            ]

            window.refresh()

            self.assertEqual(window.primary_embedded_id, 3)
            self.assertEqual(window.secondary_embedded_id, 2)
            self.assertEqual(player.sid, 3)
            self.assertEqual(player.secondary_sid, 2)
            self.assertEqual(window.overlay.snapshot.primary, "Embedded primary")
            self.assertEqual(window.overlay.snapshot.secondary, "Embedded secondary")
            window.close()

    def test_speed_and_delays_update_mpv_and_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            video = directory / "episode.mkv"
            video.touch()
            window, player = self.make_window(directory)
            window.current_video_path = str(video)

            window.speed_control.setValue(1.25)
            window.primary_offset_control.setValue(0.4)
            window.secondary_offset_control.setValue(-0.2)
            session = window.current_playback_session()

            self.assertEqual(player.speed, 1.25)
            self.assertEqual(player.sub_delay, 0.4)
            self.assertEqual(player.secondary_sub_delay, -0.2)
            self.assertEqual(session.speed, 1.25)
            self.assertEqual(session.primary_offset, 0.4)
            self.assertEqual(session.secondary_offset, -0.2)
            window.close()

    def test_hover_deduplication_includes_subtitle_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window, _player = self.make_window(Path(tmp))
            first = SubtitleInteraction("hover", "run", "primary", 5.0, "快跑")
            later = SubtitleInteraction("hover", "run", "primary", 12.0, "运行")

            window.handle_subtitle_interaction(first)
            window.handle_subtitle_interaction(first)
            self.assertEqual(window._hover_generation, 1)

            window.handle_subtitle_interaction(later)
            self.assertEqual(window._hover_generation, 2)
            window.hover_timer.stop()
            window.close()


if __name__ == "__main__":
    unittest.main()
