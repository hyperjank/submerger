#!/usr/bin/env python3
"""Simple Qt6/mpv player displaying subtitles in side columns."""
import os
os.environ['LC_NUMERIC'] = 'C'


import locale
locale.setlocale(locale.LC_NUMERIC, "C")

import sys
from typing import List
from PyQt6 import QtWidgets, QtCore, QtGui
import mpv
import pysubs2


class SubtitleColumn(QtWidgets.QListWidget):
    """A read-only list widget that highlights the active subtitle."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self._current_row = -1

    def load_subs(self, subtitle_file: str) -> None:
        subs = pysubs2.load(subtitle_file)
        for ev in subs.events:
            text = ev.plaintext.strip()
            if text:
                item = QtWidgets.QListWidgetItem(text)
                item.setData(
                    QtCore.Qt.ItemDataRole.UserRole,
                    (ev.start, ev.end),
                )
                self.addItem(item)

    def update_active(self, pos_ms: int) -> None:
        """Highlight and scroll to the subtitle covering ``pos_ms``."""
        for row in range(self.count()):
            item = self.item(row)
            start, end = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if start <= pos_ms < end:
                if row != self._current_row:
                    self._current_row = row
                    self.setCurrentRow(row)
                    self.scrollToItem(
                        item,
                        QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
                    )
                return
        self.setCurrentRow(-1)


class VideoWidget(QtWidgets.QWidget):
    """Widget hosting mpv playback."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # ensure the widget has a native window handle for mpv
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow)
        # defer player creation until the widget is shown
        self.mpv: mpv.MPV | None = None

        self._pending_load: str | None = None


    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _ensure_player(self) -> mpv.MPV:
        """Instantiate mpv using the current native window id."""
        if self.mpv is None:

            # Ensure the native window id exists on both PyQt6 and PySide6
            wid = int(self.winId())
            self.mpv = mpv.MPV(wid=wid, force_window=False)
            if self._pending_load:
                self.mpv.command("loadfile", self._pending_load)
                self._pending_load = None
        return self.mpv

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._ensure_player()


    def toggle_pause(self) -> None:
        """Toggle the pause state."""
        player = self._ensure_player()
        try:
            player.pause = not player.pause
        except AttributeError:
            # fall back to command if direct property fails
            player.command("cycle", "pause")

    def position(self) -> float:
        """Current playback position in seconds."""
        player = self._ensure_player()
        pos = player.time_pos
        return float(pos) if pos is not None else 0.0

    def duration(self) -> float:
        """Total duration in seconds."""
        player = self._ensure_player()
        dur = player.duration
        return float(dur) if dur is not None else 0.0

    def seek(self, seconds: float) -> None:
        player = self._ensure_player()
        player.command("seek", seconds, "absolute")

    def seek_relative(self, offset: float) -> None:
        """Seek relative to the current position."""
        player = self._ensure_player()
        player.command("seek", offset, "relative")

    def stop(self) -> None:
        """Stop playback and reset position."""
        player = self._ensure_player()
        player.command("stop")
        player.pause = True
    
    def load(self, path: str) -> None:
        if self.mpv is None:
            self._pending_load = path
        else:
            self.mpv.command("loadfile", path)


class PlaybackControls(QtWidgets.QWidget):
    """Playback controls with a slider."""

    positionChanged = QtCore.pyqtSignal(float)

    def __init__(self, video: VideoWidget) -> None:
        super().__init__()
        self._video = video

        self.play_button = QtWidgets.QToolButton()
        self.play_button.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MediaPause
            )
        )
        self.play_button.clicked.connect(self.toggle_play)

        self.back_button = QtWidgets.QToolButton()
        self.back_button.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MediaSeekBackward
            )
        )
        self.back_button.clicked.connect(lambda: self._video.seek_relative(-5))

        self.stop_button = QtWidgets.QToolButton()
        self.stop_button.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MediaStop
            )
        )
        self.stop_button.clicked.connect(self._video.stop)

        self.forward_button = QtWidgets.QToolButton()
        self.forward_button.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MediaSeekForward
            )
        )
        self.forward_button.clicked.connect(lambda: self._video.seek_relative(5))

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderReleased.connect(self._slider_released)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.back_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.forward_button)
        layout.addWidget(self.slider, 1)
        self.setLayout(layout)

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.timeout.connect(self.update_position)
        self._update_timer.start(500)

    def toggle_play(self) -> None:
        self._video.toggle_pause()
        self.update_button()

    def update_button(self) -> None:
        paused = bool(self._video.mpv.pause)
        icon = QtWidgets.QStyle.StandardPixmap.SP_MediaPlay if paused else QtWidgets.QStyle.StandardPixmap.SP_MediaPause
        self.play_button.setIcon(self.style().standardIcon(icon))

    def _slider_released(self) -> None:
        pos = self.slider.value() / 1000.0
        self._video.seek(pos)

    def update_position(self) -> None:
        dur = self._video.duration()
        pos = self._video.position()
        if dur:
            self.slider.blockSignals(True)
            self.slider.setRange(0, int(dur * 1000))
            self.slider.setValue(int(pos * 1000))
            self.slider.blockSignals(False)
        self.update_button()
        self.positionChanged.emit(pos)


    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self._video.mpv:
            self._video.mpv.terminate()
        super().closeEvent(event)


class PlayerWindow(QtWidgets.QMainWindow):
    def __init__(self, video: str, tl_subs: str, sl_subs: str) -> None:
        super().__init__()
        self.setWindowTitle("Bilingual Player")

        self.video_widget = VideoWidget()
        self.controls = PlaybackControls(self.video_widget)
        self.tl_list = SubtitleColumn()
        self.sl_list = SubtitleColumn()

        self.tl_list.load_subs(tl_subs)
        self.sl_list.load_subs(sl_subs)

        video_layout = QtWidgets.QVBoxLayout()
        video_layout.addWidget(self.video_widget, 1)
        video_layout.addWidget(self.controls, 0)

        video_container = QtWidgets.QWidget()
        video_container.setLayout(video_layout)

        self.setCentralWidget(video_container)

        self.tl_dock = QtWidgets.QDockWidget("TL", self)
        self.tl_dock.setWidget(self.tl_list)
        self.sl_dock = QtWidgets.QDockWidget("SL", self)
        self.sl_dock.setWidget(self.sl_list)

        for dock in (self.tl_dock, self.sl_dock):
            dock.setAllowedAreas(
                QtCore.Qt.DockWidgetArea.RightDockWidgetArea
                | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            )
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            )

        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.tl_dock)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.sl_dock)
        self.splitDockWidget(self.tl_dock, self.sl_dock, QtCore.Qt.Orientation.Vertical)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.tl_dock.toggleViewAction())
        view_menu.addAction(self.sl_dock.toggleViewAction())

        self.video_widget.load(video)

        self.controls.positionChanged.connect(self._update_subs)

        QtGui.QShortcut(QtGui.QKeySequence("Left"), self).activated.connect(
            lambda: self.video_widget.seek_relative(-5)
        )
        QtGui.QShortcut(QtGui.QKeySequence("Right"), self).activated.connect(
            lambda: self.video_widget.seek_relative(5)
        )

        QtGui.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(
            self.controls.toggle_play
        )

    def _update_subs(self, pos_s: float) -> None:
        pos_ms = int(pos_s * 1000)
        self.tl_list.update_active(pos_ms)
        self.sl_list.update_active(pos_ms)


def main(args: List[str]) -> int:
    if len(args) != 4:
        print("Usage: qt_mpv_scroll.py video.mp4 tl.srt sl.srt")
        return 1
    _, video, tl_subs, sl_subs = args

    app = QtWidgets.QApplication([args[0]])
    window = PlayerWindow(video, tl_subs, sl_subs)
    window.resize(1024, 600)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
