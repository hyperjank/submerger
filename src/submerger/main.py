from __future__ import annotations

import argparse
import locale
import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication, QStyleFactory


def prepare_process_environment() -> None:
    locale.setlocale(locale.LC_NUMERIC, "C")

    requested_style = os.environ.get("QT_STYLE_OVERRIDE")
    if requested_style and requested_style not in QStyleFactory.keys():
        os.environ.pop("QT_STYLE_OVERRIDE", None)


def configure_opengl() -> None:
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bilingual mpv video player")
    parser.add_argument("video", nargs="?", help="Episode video to open.")
    parser.add_argument("--primary", help="Primary external SRT subtitle.")
    parser.add_argument("--secondary", help="Secondary external SRT subtitle.")
    parser.add_argument("--alignment", help="Alignment sidecar to load.")
    parser.add_argument("--no-restore", action="store_true", help="Start without restoring the last episode.")
    return parser


def parse_cli_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_cli_arguments(sys.argv[1:] if arguments is None else arguments)
    prepare_process_environment()
    configure_opengl()

    app = QApplication([sys.argv[0]])
    prepare_process_environment()

    from .theme import apply_application_theme

    apply_application_theme(app)

    from .player import MainWindow

    window = MainWindow(restore_session=not args.no_restore and args.video is None)
    window.show()
    if args.video:
        QTimer.singleShot(
            0,
            lambda: window.open_episode(
                args.video,
                primary_path=args.primary,
                secondary_path=args.secondary,
                alignment_path=args.alignment,
            ),
        )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
