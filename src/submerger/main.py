from __future__ import annotations

import locale
import os
import sys

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


def main() -> int:
    prepare_process_environment()
    configure_opengl()

    app = QApplication(sys.argv)
    prepare_process_environment()

    from .player import MainWindow

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
