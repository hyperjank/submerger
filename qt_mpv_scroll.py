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
    """A read-only list widget to display subtitle lines."""

    def load_subs(self, subtitle_file: str) -> None:
        subs = pysubs2.load(subtitle_file)
        for ev in subs.events:
            text = ev.plaintext.strip()
            if text:
                item = QtWidgets.QListWidgetItem(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, (ev.start, ev.end))
                self.addItem(item)


class VideoWidget(QtWidgets.QWidget):
    """Widget hosting mpv playback."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # ensure the widget has a native window handle for mpv
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow)
        self.mpv = mpv.MPV(wid=int(self.winId()))

    def load(self, path: str) -> None:
        self.mpv.command("loadfile", path)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self.mpv.terminate()
        super().closeEvent(event)


class PlayerWindow(QtWidgets.QMainWindow):
    def __init__(self, video: str, tl_subs: str, sl_subs: str) -> None:
        super().__init__()
        self.setWindowTitle("Bilingual Player")

        self.video_widget = VideoWidget()
        self.tl_list = SubtitleColumn()
        self.sl_list = SubtitleColumn()

        self.tl_list.load_subs(tl_subs)
        self.sl_list.load_subs(sl_subs)

        # layout: video on the left, two subtitle columns on the right
        subs_layout = QtWidgets.QHBoxLayout()
        subs_layout.addWidget(self.tl_list)
        subs_layout.addWidget(self.sl_list)

        subs_container = QtWidgets.QWidget()
        subs_container.setLayout(subs_layout)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.video_widget, 2)
        main_layout.addWidget(subs_container, 1)

        central = QtWidgets.QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.video_widget.load(video)


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
