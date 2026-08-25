from __future__ import annotations

import html
import locale
import logging
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCursor, QKeySequence, QOpenGLContext
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedLayout,
    QStyle,
    QTextBrowser,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
import mpv

from .alignment_panel import AlignmentPanel, AlignmentWorker
from .alignment import load_alignment_package, tracks_from_alignment_package
from .interaction import SubtitleInteraction
from .plugins import PluginContext, PluginRegistry, PluginResult, create_default_registry
from .playback import (
    PlaybackSession,
    PlaybackSessionStore,
    VIDEO_EXTENSIONS,
    classify_dropped_paths,
    discover_alignment_sidecar,
    discover_external_subtitles,
    is_text_subtitle_track,
    language_matches,
    subtitle_track_label,
)
from .script_sidebar import ScriptSidebar
from .settings import LLMEndpointSettings, load_llm_settings, save_llm_settings
from .settings_dialog import LLMSettingsDialog
from .subtitle_display import SubtitleOverlay
from .subtitles import DualSubtitleEngine


LOGGER = logging.getLogger(__name__)


class PluginWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class PluginWorker(QRunnable):
    def __init__(self, registry: PluginRegistry, plugin_id: str, action: str, context: PluginContext) -> None:
        super().__init__()
        self.registry = registry
        self.plugin_id = plugin_id
        self.action = action
        self.context = context
        self.signals = PluginWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.registry.run(self.plugin_id, self.action, self.context))
        except Exception as exc:  # noqa: BLE001 - surface plugin failures in the dock.
            self.signals.failed.emit(str(exc))


class DockTitleBar(QWidget):
    def __init__(self, title: str, dock: QDockWidget) -> None:
        super().__init__(dock)
        self.label = QLabel(title)
        self.float_button = QToolButton()
        self.float_button.setText("□")
        self.close_button = QToolButton()
        self.close_button.setText("×")
        self.float_button.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        self.close_button.clicked.connect(dock.hide)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.float_button)
        layout.addWidget(self.close_button)
        self.setStyleSheet(
            """
            DockTitleBar {
                background: #111827;
                color: #e5e7eb;
            }
            QLabel {
                color: #e5e7eb;
                font-weight: 600;
            }
            QToolButton {
                background: #1f2937;
                color: #e5e7eb;
                border: 1px solid #374151;
                border-radius: 3px;
                min-width: 18px;
                min-height: 18px;
            }
            """
        )


class MpvVideoWidget(QOpenGLWidget):
    render_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.player: mpv.MPV | None = None
        self.render_context: mpv.MpvRenderContext | None = None
        self._get_proc_address_callback = None
        self._needs_render = False
        self.render_requested.connect(self.update, Qt.ConnectionType.QueuedConnection)
        self.setStyleSheet("background: #050505;")

    def initializeGL(self) -> None:
        pass

    def resizeGL(self, _width: int, _height: int) -> None:
        self.update()

    def paintGL(self) -> None:
        if self.render_context is None:
            return

        ratio = self.devicePixelRatioF()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))
        self.render_context.update()
        self._needs_render = False
        self.render_context.render(
            opengl_fbo={
                "w": width,
                "h": height,
                "fbo": self.defaultFramebufferObject(),
            },
            flip_y=True,
        )
        self.render_context.report_swap()

    def _ensure_player(self) -> mpv.MPV:
        if self.player is not None:
            return self.player

        locale.setlocale(locale.LC_NUMERIC, "C")
        self.player = mpv.MPV(
            vo="libmpv",
            input_default_bindings=True,
            input_vo_keyboard=True,
            osc=False,
            sub_auto="no",
            sid="no",
            keep_open=True,
        )
        self._get_proc_address_callback = mpv.MpvGlGetProcAddressFn(self._get_proc_address)
        self.render_context = mpv.MpvRenderContext(
            self.player,
            "opengl",
            opengl_init_params={"get_proc_address": self._get_proc_address_callback},
        )
        self.render_context.update_cb = self._on_mpv_render_update
        return self.player

    def _get_proc_address(self, _ctx, name: bytes) -> int:
        context = QOpenGLContext.currentContext()
        if context is None:
            LOGGER.debug("No current OpenGL context while resolving %r", name)
            return 0

        try:
            address = context.getProcAddress(name)
        except TypeError:
            address = context.getProcAddress(name.decode("ascii"))
        return int(address or 0)

    def _on_mpv_render_update(self) -> None:
        self._needs_render = True
        self.render_requested.emit()

    def shutdown(self) -> None:
        self.makeCurrent()
        try:
            if self.render_context is not None:
                self.render_context.update_cb = None
                self.render_context.free()
                self.render_context = None
                self._get_proc_address_callback = None
            if self.player is not None:
                self.player.terminate()
                self.player = None
        finally:
            self.doneCurrent()


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        session_store: PlaybackSessionStore | None = None,
        restore_session: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Submerger")
        self.resize(1100, 720)
        self.setAcceptDrops(True)

        self.subtitle_engine = DualSubtitleEngine()
        self.llm_settings = load_llm_settings()
        self.plugin_registry = create_default_registry(llm_settings=self.llm_settings)
        self.thread_pool = QThreadPool.globalInstance()
        self.duration = 0.0
        self._seeking = False
        self._last_hover_text = ""
        self.current_plugin_context: PluginContext | None = None
        self.plugin_action_buttons: list[QPushButton] = []
        self.primary_subtitle_path: str | None = None
        self.secondary_subtitle_path: str | None = None
        self.alignment_sidecar_path: str | None = None
        self.current_video_path: str | None = None
        self.primary_offset = 0.0
        self.secondary_offset = 0.0
        self.playback_speed = 1.0
        self.primary_embedded_id: int | None = None
        self.secondary_embedded_id: int | None = None
        self._pending_primary_embedded_id: int | None = None
        self._pending_secondary_embedded_id: int | None = None
        self._pending_seek_position: float | None = None
        self._track_signature: tuple | None = None
        self._last_session_save_second = -1
        self._pre_fullscreen_docks: dict[QDockWidget, bool] = {}
        self.session_store = session_store or PlaybackSessionStore()

        self.video_surface = MpvVideoWidget()
        self.overlay = SubtitleOverlay()

        self.open_video_button = QPushButton("Open Episode")
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.previous_line_button = QPushButton()
        self.previous_line_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.previous_line_button.setToolTip("Previous subtitle line (Ctrl+Left)")
        self.replay_line_button = QPushButton("Replay")
        self.replay_line_button.setToolTip("Replay current subtitle line (R)")
        self.next_line_button = QPushButton()
        self.next_line_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self.next_line_button.setToolTip("Next subtitle line (Ctrl+Right)")
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_label = QLabel("00:00 / 00:00")
        self.speed_control = QDoubleSpinBox()
        self.speed_control.setRange(0.25, 3.0)
        self.speed_control.setSingleStep(0.05)
        self.speed_control.setDecimals(2)
        self.speed_control.setValue(1.0)
        self.speed_control.setSuffix("×")
        self.speed_control.setToolTip("Playback speed ([ and ], Backspace resets)")
        self.primary_offset_control = subtitle_offset_control("Primary subtitle delay")
        self.secondary_offset_control = subtitle_offset_control("Secondary subtitle delay")
        self.fullscreen_button = QPushButton("⛶")
        self.fullscreen_button.setToolTip("Fullscreen (F11)")
        self.plugin_output = QTextBrowser()
        self.plugin_actions_bar = QHBoxLayout()
        self.plugin_panel = QWidget()
        self.plugin_dock = QDockWidget("Language Tools", self)
        self.script_button = QPushButton("Script")
        self.script_sidebar = ScriptSidebar()
        self.script_dock = QDockWidget("Episode Script", self)
        self.align_button = QPushButton("Align")
        self.alignment_panel = AlignmentPanel()
        self.alignment_panel.apply_llm_settings(self.llm_settings)
        self.alignment_dock = QDockWidget("Subtitle Alignment", self)

        self.recent_menu: QMenu | None = None
        self.primary_embedded_menu: QMenu | None = None
        self.secondary_embedded_menu: QMenu | None = None
        self.controls_widget: QWidget | None = None

        self._build_layout()
        self._build_menu()
        self._connect()

        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        if restore_session:
            QTimer.singleShot(0, self.restore_last_session)

    def _ensure_player(self) -> mpv.MPV:
        self.video_surface.makeCurrent()
        try:
            return self.video_surface._ensure_player()
        finally:
            self.video_surface.doneCurrent()

    def _build_layout(self) -> None:
        stage = QFrame()
        stage.setFrameShape(QFrame.Shape.NoFrame)
        stack = QStackedLayout(stage)
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        stack.addWidget(self.video_surface)
        stack.addWidget(self.overlay)
        stack.setCurrentWidget(self.overlay)
        self.overlay.raise_()

        controls = QHBoxLayout()
        controls.setContentsMargins(10, 8, 10, 8)
        controls.setSpacing(8)
        controls.addWidget(self.open_video_button)
        controls.addWidget(self.script_button)
        controls.addWidget(self.align_button)
        controls.addWidget(self.previous_line_button)
        controls.addWidget(self.replay_line_button)
        controls.addWidget(self.next_line_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.position_slider, 1)
        controls.addWidget(self.time_label)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.speed_control)
        controls.addWidget(QLabel("P"))
        controls.addWidget(self.primary_offset_control)
        controls.addWidget(QLabel("S"))
        controls.addWidget(self.secondary_offset_control)
        controls.addWidget(self.fullscreen_button)

        self.controls_widget = QWidget()
        self.controls_widget.setLayout(controls)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(stage, 1)
        root.addWidget(self.controls_widget)

        container = QWidget()
        container.setObjectName("root")
        container.setLayout(root)
        container.setStyleSheet(
            """
            QWidget#root { background: #111827; color: #e5e7eb; }
            QPushButton {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                color: #f9fafb;
                min-height: 30px;
                padding: 0 12px;
            }
            QPushButton:hover { background: #273244; }
            QLabel { color: #d1d5db; }
            QDoubleSpinBox {
                background: #0f172a;
                border: 1px solid #374151;
                border-radius: 4px;
                color: #f9fafb;
                min-height: 28px;
                max-width: 76px;
                padding: 0 4px;
            }
            SubtitleOverlay { background: transparent; }
            """
        )
        self.setCentralWidget(container)

        self.plugin_output.setReadOnly(True)
        self.plugin_output.setPlainText("Hover a word for dictionary context. Drag-select a phrase for explanation.")
        self.plugin_output.setStyleSheet(
            """
            QTextBrowser {
                background: #0f172a;
                color: #e2e8f0;
                border: 0;
                font-size: 13px;
                padding: 10px;
            }
            """
        )
        self.plugin_actions_bar.setContentsMargins(8, 8, 8, 0)
        self.plugin_actions_bar.addStretch(1)
        plugin_layout = QVBoxLayout(self.plugin_panel)
        plugin_layout.setContentsMargins(0, 0, 0, 0)
        plugin_layout.setSpacing(0)
        plugin_layout.addLayout(self.plugin_actions_bar)
        plugin_layout.addWidget(self.plugin_output, 1)
        self.plugin_dock.setWidget(self.plugin_panel)
        self.plugin_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.plugin_dock.setTitleBarWidget(DockTitleBar("Language Tools", self.plugin_dock))
        self.plugin_dock.setMinimumWidth(300)
        self.plugin_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.plugin_dock)

        self.script_dock.setWidget(self.script_sidebar)
        self.script_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.script_dock.setTitleBarWidget(DockTitleBar("Episode Script", self.script_dock))
        self.script_dock.setMinimumWidth(300)
        self.script_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.script_dock)

        self.alignment_dock.setWidget(self.alignment_panel)
        self.alignment_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.alignment_dock.setTitleBarWidget(DockTitleBar("Subtitle Alignment", self.alignment_dock))
        self.alignment_dock.setMinimumWidth(360)
        self.alignment_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.alignment_dock)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.make_action("Open Episode…", "Ctrl+O", self.open_video))
        file_menu.addAction(self.make_action("Open Primary Subtitle…", None, lambda: self.open_subtitle(primary=True)))
        file_menu.addAction(self.make_action("Open Secondary Subtitle…", None, lambda: self.open_subtitle(primary=False)))
        file_menu.addAction(self.make_action("Open Alignment…", None, self.open_alignment_sidecar))
        self.recent_menu = file_menu.addMenu("Recent Episodes")
        file_menu.addSeparator()
        file_menu.addAction(self.make_action("Quit", "Ctrl+Q", self.close))

        playback_menu = self.menuBar().addMenu("Playback")
        playback_menu.addAction(self.make_action("Play/Pause", "Space", self.toggle_playback))
        playback_menu.addAction(self.make_action("Previous Subtitle Line", "Ctrl+Left", lambda: self.navigate_subtitle(-1)))
        playback_menu.addAction(self.make_action("Replay Current Line", "R", self.replay_current_line))
        playback_menu.addAction(self.make_action("Next Subtitle Line", "Ctrl+Right", lambda: self.navigate_subtitle(1)))
        playback_menu.addSeparator()
        playback_menu.addAction(self.make_action("Seek Back 5 Seconds", "Left", lambda: self.seek_relative(-5)))
        playback_menu.addAction(self.make_action("Seek Forward 5 Seconds", "Right", lambda: self.seek_relative(5)))
        playback_menu.addSeparator()
        playback_menu.addAction(self.make_action("Slower", "[", lambda: self.adjust_speed(-0.1)))
        playback_menu.addAction(self.make_action("Faster", "]", lambda: self.adjust_speed(0.1)))
        playback_menu.addAction(self.make_action("Normal Speed", "Backspace", lambda: self.speed_control.setValue(1.0)))

        subtitle_menu = self.menuBar().addMenu("Subtitles")
        subtitle_menu.addAction(self.make_action("Open Primary Subtitle…", None, lambda: self.open_subtitle(primary=True)))
        subtitle_menu.addAction(self.make_action("Open Secondary Subtitle…", None, lambda: self.open_subtitle(primary=False)))
        subtitle_menu.addSeparator()
        self.primary_embedded_menu = subtitle_menu.addMenu("Primary Embedded Track")
        self.secondary_embedded_menu = subtitle_menu.addMenu("Secondary Embedded Track")
        subtitle_menu.addSeparator()
        subtitle_menu.addAction(self.make_action("Primary Earlier 0.1s", "Z", lambda: self.adjust_subtitle_offset(True, -0.1)))
        subtitle_menu.addAction(self.make_action("Primary Later 0.1s", "X", lambda: self.adjust_subtitle_offset(True, 0.1)))
        subtitle_menu.addAction(self.make_action("Secondary Earlier 0.1s", "Shift+Z", lambda: self.adjust_subtitle_offset(False, -0.1)))
        subtitle_menu.addAction(self.make_action("Secondary Later 0.1s", "Shift+X", lambda: self.adjust_subtitle_offset(False, 0.1)))
        subtitle_menu.addAction(self.make_action("Reset Subtitle Delays", "Ctrl+0", self.reset_subtitle_offsets))

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.make_action("Episode Script", None, self.toggle_script_sidebar))
        view_menu.addAction(self.make_action("Subtitle Alignment", None, self.toggle_alignment_panel))
        view_menu.addAction(self.make_action("Toggle Fullscreen", "F11", self.toggle_fullscreen))
        view_menu.addAction(self.make_action("Exit Fullscreen", "Escape", self.exit_fullscreen))

        settings_menu = self.menuBar().addMenu("Settings")
        llm_action = QAction("LLM Endpoint Settings", self)
        llm_action.triggered.connect(self.open_llm_settings)
        settings_menu.addAction(llm_action)
        self.update_recent_menu()

    def make_action(self, text: str, shortcut: str | None, callback) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.centralWidget().addAction(action)
        action.triggered.connect(callback)
        return action

    def _connect(self) -> None:
        self.open_video_button.clicked.connect(self.open_video)
        self.play_button.clicked.connect(self.toggle_playback)
        self.previous_line_button.clicked.connect(lambda: self.navigate_subtitle(-1))
        self.replay_line_button.clicked.connect(self.replay_current_line)
        self.next_line_button.clicked.connect(lambda: self.navigate_subtitle(1))
        self.speed_control.valueChanged.connect(self.set_playback_speed)
        self.primary_offset_control.valueChanged.connect(lambda value: self.set_subtitle_offset(True, value))
        self.secondary_offset_control.valueChanged.connect(lambda value: self.set_subtitle_offset(False, value))
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.script_button.clicked.connect(self.toggle_script_sidebar)
        self.align_button.clicked.connect(self.toggle_alignment_panel)
        self.script_sidebar.seek_requested.connect(self.seek_to)
        self.alignment_panel.run_requested.connect(self.run_alignment)
        self.alignment_panel.load_alignment_requested.connect(self.load_alignment_sidecar)
        self.position_slider.sliderPressed.connect(self._begin_seek)
        self.position_slider.sliderReleased.connect(self._end_seek)
        self.overlay.interaction_requested.connect(self.handle_subtitle_interaction)
        self.overlay.fullscreen_requested.connect(self.toggle_fullscreen)

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open episode",
            str(Path.home()),
            "Video files (*.mkv *.mp4 *.avi *.mov *.webm);;All files (*)",
        )
        if path:
            self.open_episode(path)

    def open_episode(
        self,
        video_path: str | Path,
        *,
        primary_path: str | Path | None = None,
        secondary_path: str | Path | None = None,
        alignment_path: str | Path | None = None,
        position: float = 0.0,
        speed: float = 1.0,
        primary_offset: float = 0.0,
        secondary_offset: float = 0.0,
        primary_embedded_id: int | None = None,
        secondary_embedded_id: int | None = None,
    ) -> None:
        video = Path(video_path).expanduser().resolve()
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            QMessageBox.critical(self, "Episode error", f"Video file not found or unsupported:\n{video}")
            return

        self.save_current_session()
        self.current_video_path = str(video)
        self.primary_subtitle_path = None
        self.secondary_subtitle_path = None
        self.alignment_sidecar_path = None
        self.subtitle_engine = DualSubtitleEngine()
        self.duration = 0.0
        self._last_session_save_second = -1
        self.script_sidebar.set_tracks(self.subtitle_engine.primary, self.subtitle_engine.secondary)
        self._track_signature = None
        self.primary_embedded_id = None
        self.secondary_embedded_id = None
        self._pending_primary_embedded_id = primary_embedded_id
        self._pending_secondary_embedded_id = secondary_embedded_id
        self._pending_seek_position = max(0.0, float(position))

        detected_primary, detected_secondary = discover_external_subtitles(
            video,
            primary_language=self.alignment_panel.primary_language.text().strip() or "en",
            secondary_language=self.alignment_panel.secondary_language.text().strip() or "zh",
        )
        resolved_alignment = (
            Path(alignment_path).expanduser()
            if alignment_path
            else discover_alignment_sidecar(video)
            if primary_path is None and secondary_path is None
            else None
        )
        if resolved_alignment and resolved_alignment.is_file():
            self.load_alignment_sidecar(str(resolved_alignment))
        else:
            resolved_primary = Path(primary_path).expanduser() if primary_path else detected_primary
            resolved_secondary = Path(secondary_path).expanduser() if secondary_path else detected_secondary
            if resolved_primary and resolved_primary.is_file():
                self.load_subtitle_path(resolved_primary, primary=True)
            if resolved_secondary and resolved_secondary.is_file():
                self.load_subtitle_path(resolved_secondary, primary=False)

        self.speed_control.setValue(speed)
        self.primary_offset_control.setValue(primary_offset)
        self.secondary_offset_control.setValue(secondary_offset)
        player = self._ensure_player()
        player.sub_visibility = False
        player.secondary_sub_visibility = False
        player.sid = "no"
        player.secondary_sid = "no"
        player.command("loadfile", str(video))
        player.speed = self.playback_speed
        player.sub_delay = self.primary_offset
        player.secondary_sub_delay = self.secondary_offset
        player.pause = False
        self.setWindowTitle(f"{video.name} — Submerger")
        self.save_current_session()
        self.update_recent_menu()

    def open_playback_session(self, session: PlaybackSession) -> None:
        self.open_episode(
            session.video_path,
            primary_path=session.primary_subtitle_path,
            secondary_path=session.secondary_subtitle_path,
            alignment_path=session.alignment_sidecar_path,
            position=session.position,
            speed=session.speed,
            primary_offset=session.primary_offset,
            secondary_offset=session.secondary_offset,
            primary_embedded_id=session.primary_embedded_id,
            secondary_embedded_id=session.secondary_embedded_id,
        )

    def open_subtitle(self, *, primary: bool) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open subtitle",
            str(Path.home()),
            "SubRip subtitles (*.srt);;All files (*)",
        )
        if not path:
            return

        self.load_subtitle_path(path, primary=primary)

    def load_subtitle_path(self, path: str | Path, *, primary: bool) -> None:
        resolved = Path(path).expanduser().resolve()

        try:
            if primary:
                self.subtitle_engine.load_primary(resolved)
                self.primary_subtitle_path = str(resolved)
                self.select_embedded_track(primary=True, track_id=None, clear_external=False)
            else:
                self.subtitle_engine.load_secondary(resolved)
                self.secondary_subtitle_path = str(resolved)
                self.select_embedded_track(primary=False, track_id=None, clear_external=False)
            self.alignment_sidecar_path = None
            self.script_sidebar.set_tracks(self.subtitle_engine.primary, self.subtitle_engine.secondary)
            self.alignment_panel.set_loaded_paths(
                self.primary_subtitle_path,
                self.secondary_subtitle_path,
                clear_missing=True,
            )
        except Exception as exc:  # noqa: BLE001 - show parser/file failures to the user.
            QMessageBox.critical(self, "Subtitle error", str(exc))

    def open_alignment_sidecar(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open alignment sidecar",
            str(Path.home()),
            "Submerger alignment (*.alignment.json);;JSON files (*.json)",
        )
        if path:
            self.load_alignment_sidecar(path)

    def toggle_playback(self) -> None:
        player = self._ensure_player()
        player.pause = not bool(player.pause)

    def refresh(self) -> None:
        player = self.video_surface.player
        if player is None:
            self.overlay.set_subtitles("", "", None)
            return

        try:
            timestamp = player.time_pos
            self.duration = float(player.duration or self.duration or 0.0)
        except Exception:  # noqa: BLE001 - properties are temporarily unavailable while loading.
            return

        self.refresh_embedded_subtitle_tracks()
        if self._pending_seek_position is not None and self.duration > 0:
            position = min(self._pending_seek_position, max(0.0, self.duration - 0.25))
            self._pending_seek_position = None
            self.seek_to(position)

        primary, secondary = self.subtitle_engine.active(
            timestamp,
            self.primary_offset,
            self.secondary_offset,
        )
        if self.primary_subtitle_path is None and self.primary_embedded_id is not None:
            primary = safe_mpv_text(player, "sub_text")
        if self.secondary_subtitle_path is None and self.secondary_embedded_id is not None:
            secondary = safe_mpv_text(player, "secondary_sub_text")
        self.overlay.set_subtitles(primary, secondary, timestamp)
        script_offset = self.primary_offset if self.subtitle_engine.primary.cues else self.secondary_offset
        script_timestamp = None if timestamp is None else timestamp - script_offset
        self.script_sidebar.update_position(script_timestamp)

        if not self._seeking and self.duration > 0 and timestamp is not None:
            self.position_slider.setRange(0, int(self.duration * 1000))
            self.position_slider.setValue(int(float(timestamp) * 1000))

        self.time_label.setText(f"{format_time(timestamp)} / {format_time(self.duration)}")
        icon = QStyle.StandardPixmap.SP_MediaPlay if player.pause else QStyle.StandardPixmap.SP_MediaPause
        self.play_button.setIcon(self.style().standardIcon(icon))
        if timestamp is not None and self.current_video_path:
            whole_second = int(timestamp)
            if whole_second > 0 and whole_second % 5 == 0 and whole_second != self._last_session_save_second:
                self._last_session_save_second = whole_second
                self.save_current_session()

    def _begin_seek(self) -> None:
        self._seeking = True

    def _end_seek(self) -> None:
        self._seeking = False
        player = self.video_surface.player
        if player is not None:
            self.seek_to(self.position_slider.value() / 1000.0)

    def seek_relative(self, seconds: float) -> None:
        player = self.video_surface.player
        if player is not None:
            player.seek(seconds, reference="relative", precision="exact")

    def replay_current_line(self) -> None:
        player = self.video_surface.player
        if player is None:
            return
        timestamp = player.time_pos
        track, offset, embedded_role = self.navigation_source()
        if track.cues:
            cue = track.cue_at_or_before(None if timestamp is None else timestamp - offset)
            if cue is not None:
                self.seek_to(cue.start + offset)
                player.pause = False
            return
        if embedded_role:
            player.command("sub-seek", 0, embedded_role)
            player.pause = False

    def navigate_subtitle(self, direction: int) -> None:
        player = self.video_surface.player
        if player is None or direction == 0:
            return
        timestamp = player.time_pos
        track, offset, embedded_role = self.navigation_source()
        if track.cues:
            cue = track.adjacent_cue(None if timestamp is None else timestamp - offset, direction)
            if cue is not None:
                self.seek_to(cue.start + offset)
            return
        if embedded_role:
            player.command("sub-seek", 1 if direction > 0 else -1, embedded_role)

    def navigation_source(self):
        if self.subtitle_engine.primary.cues:
            return self.subtitle_engine.primary, self.primary_offset, None
        if self.subtitle_engine.secondary.cues:
            return self.subtitle_engine.secondary, self.secondary_offset, None
        if self.primary_embedded_id is not None:
            return self.subtitle_engine.primary, self.primary_offset, "primary"
        if self.secondary_embedded_id is not None:
            return self.subtitle_engine.secondary, self.secondary_offset, "secondary"
        return self.subtitle_engine.primary, 0.0, None

    def set_playback_speed(self, value: float) -> None:
        self.playback_speed = round(float(value), 2)
        player = self.video_surface.player
        if player is not None:
            player.speed = self.playback_speed

    def adjust_speed(self, delta: float) -> None:
        self.speed_control.setValue(round(self.speed_control.value() + delta, 2))

    def set_subtitle_offset(self, primary: bool, value: float) -> None:
        offset = round(float(value), 1)
        player = self.video_surface.player
        if primary:
            self.primary_offset = offset
            if player is not None:
                player.sub_delay = offset
        else:
            self.secondary_offset = offset
            if player is not None:
                player.secondary_sub_delay = offset

    def adjust_subtitle_offset(self, primary: bool, delta: float) -> None:
        control = self.primary_offset_control if primary else self.secondary_offset_control
        control.setValue(round(control.value() + delta, 1))

    def reset_subtitle_offsets(self) -> None:
        self.primary_offset_control.setValue(0.0)
        self.secondary_offset_control.setValue(0.0)

    def refresh_embedded_subtitle_tracks(self) -> None:
        player = self.video_surface.player
        if player is None or self.current_video_path is None:
            return
        try:
            tracks = [track for track in (player.track_list or []) if track.get("type") == "sub" and not track.get("external")]
        except Exception:  # noqa: BLE001 - track metadata is unavailable during file loading.
            return
        signature = tuple(
            (track.get("id"), track.get("lang"), track.get("title"), track.get("codec"), track.get("default"))
            for track in tracks
        )
        if signature == self._track_signature:
            return
        self._track_signature = signature

        available_ids = {int(track["id"]) for track in tracks if isinstance(track.get("id"), int)}
        if self.primary_subtitle_path is None:
            requested = self._pending_primary_embedded_id
            selected = requested if requested in available_ids else preferred_embedded_track(
                tracks,
                self.alignment_panel.primary_language.text(),
                excluded=None,
            )
            self.select_embedded_track(primary=True, track_id=selected, clear_external=False)
        if self.secondary_subtitle_path is None:
            requested = self._pending_secondary_embedded_id
            selected = requested if requested in available_ids else preferred_embedded_track(
                tracks,
                self.alignment_panel.secondary_language.text(),
                excluded=self.primary_embedded_id,
            )
            self.select_embedded_track(primary=False, track_id=selected, clear_external=False)
        self._pending_primary_embedded_id = None
        self._pending_secondary_embedded_id = None
        self.populate_embedded_track_menu(self.primary_embedded_menu, tracks, primary=True)
        self.populate_embedded_track_menu(self.secondary_embedded_menu, tracks, primary=False)

    def populate_embedded_track_menu(self, menu: QMenu | None, tracks: list[dict], *, primary: bool) -> None:
        if menu is None:
            return
        menu.clear()
        group = QActionGroup(menu)
        group.setExclusive(True)
        selected_id = self.primary_embedded_id if primary else self.secondary_embedded_id
        off = menu.addAction("Off / use external SRT")
        off.setCheckable(True)
        off.setChecked(selected_id is None)
        off.triggered.connect(lambda _checked=False, role=primary: self.select_embedded_track(primary=role, track_id=None))
        group.addAction(off)
        if tracks:
            menu.addSeparator()
        for track in tracks:
            track_id = track.get("id")
            if not isinstance(track_id, int):
                continue
            action = menu.addAction(subtitle_track_label(track))
            action.setCheckable(True)
            action.setChecked(track_id == selected_id)
            action.setEnabled(is_text_subtitle_track(track))
            if not action.isEnabled():
                action.setText(f"{action.text()} · image subtitles unsupported")
            action.triggered.connect(
                lambda _checked=False, role=primary, value=track_id: self.select_embedded_track(primary=role, track_id=value)
            )
            group.addAction(action)

    def select_embedded_track(self, *, primary: bool, track_id: int | None, clear_external: bool = True) -> None:
        player = self.video_surface.player
        value = "no" if track_id is None else track_id
        if primary:
            if track_id is not None and track_id == self.secondary_embedded_id:
                self.secondary_embedded_id = None
                if player is not None:
                    player.secondary_sid = "no"
            self.primary_embedded_id = track_id
            if player is not None:
                player.sid = value
            if clear_external and track_id is not None:
                self.subtitle_engine.primary = type(self.subtitle_engine.primary)()
                self.primary_subtitle_path = None
        else:
            if track_id is not None and track_id == self.primary_embedded_id:
                self.primary_embedded_id = None
                if player is not None:
                    player.sid = "no"
            self.secondary_embedded_id = track_id
            if player is not None:
                player.secondary_sid = value
            if clear_external and track_id is not None:
                self.subtitle_engine.secondary = type(self.subtitle_engine.secondary)()
                self.secondary_subtitle_path = None
        if clear_external and track_id is not None:
            self.alignment_sidecar_path = None
            self.script_sidebar.set_tracks(self.subtitle_engine.primary, self.subtitle_engine.secondary)
            self.alignment_panel.set_loaded_paths(
                self.primary_subtitle_path,
                self.secondary_subtitle_path,
                clear_missing=True,
            )
        if self._track_signature:
            try:
                tracks = [track for track in (player.track_list or []) if track.get("type") == "sub" and not track.get("external")] if player else []
            except Exception:  # noqa: BLE001
                tracks = []
            self.populate_embedded_track_menu(self.primary_embedded_menu, tracks, primary=True)
            self.populate_embedded_track_menu(self.secondary_embedded_menu, tracks, primary=False)

    def current_playback_session(self) -> PlaybackSession | None:
        if self.current_video_path is None:
            return None
        player = self.video_surface.player
        try:
            current_position = (
                self._pending_seek_position
                if self._pending_seek_position is not None
                else player.time_pos
                if player is not None
                else None
            )
            position = float(
                current_position
                if current_position is not None
                else self._pending_seek_position or 0.0
            )
        except Exception:  # noqa: BLE001
            position = float(self._pending_seek_position or 0.0)
        return PlaybackSession(
            video_path=self.current_video_path,
            primary_subtitle_path=self.primary_subtitle_path,
            secondary_subtitle_path=self.secondary_subtitle_path,
            alignment_sidecar_path=self.alignment_sidecar_path,
            position=position,
            speed=self.playback_speed,
            primary_offset=self.primary_offset,
            secondary_offset=self.secondary_offset,
            primary_embedded_id=self.primary_embedded_id,
            secondary_embedded_id=self.secondary_embedded_id,
        )

    def save_current_session(self) -> None:
        session = self.current_playback_session()
        if session is not None:
            try:
                self.session_store.remember(session)
            except OSError as exc:
                LOGGER.warning("Could not save playback session: %s", exc)

    def restore_last_session(self) -> None:
        session = self.session_store.last_session()
        if session is not None:
            self.open_playback_session(session)

    def update_recent_menu(self) -> None:
        if self.recent_menu is None:
            return
        self.recent_menu.clear()
        sessions = self.session_store.sessions()
        if not sessions:
            empty = self.recent_menu.addAction("No recent episodes")
            empty.setEnabled(False)
            return
        for session in sessions:
            action = self.recent_menu.addAction(session.title)
            action.setToolTip(session.video_path)
            action.triggered.connect(lambda _checked=False, item=session: self.open_playback_session(item))

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.exit_fullscreen()
            return
        self._pre_fullscreen_docks = {
            dock: dock.isVisible()
            for dock in (self.plugin_dock, self.script_dock, self.alignment_dock)
        }
        for dock in self._pre_fullscreen_docks:
            dock.hide()
        self.menuBar().hide()
        if self.controls_widget is not None:
            self.controls_widget.hide()
        self.showFullScreen()

    def exit_fullscreen(self) -> None:
        if not self.isFullScreen():
            return
        self.showNormal()
        self.menuBar().show()
        if self.controls_widget is not None:
            self.controls_widget.show()
        for dock, was_visible in self._pre_fullscreen_docks.items():
            dock.setVisible(was_visible)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls() and classify_dropped_paths(
            [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        ) != {"video": [], "subtitle": [], "alignment": []}:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        dropped = classify_dropped_paths(paths)
        primary_language = self.alignment_panel.primary_language.text()
        secondary_language = self.alignment_panel.secondary_language.text()
        subtitles = dropped["subtitle"]
        primary = next((path for path in subtitles if language_matches(path, primary_language)), None)
        secondary = next((path for path in subtitles if path != primary and language_matches(path, secondary_language)), None)
        remaining = [path for path in subtitles if path not in {primary, secondary}]
        primary = primary or (remaining.pop(0) if remaining else None)
        secondary = secondary or (remaining.pop(0) if remaining else None)
        if dropped["video"]:
            self.open_episode(
                dropped["video"][0],
                primary_path=primary,
                secondary_path=secondary,
                alignment_path=dropped["alignment"][0] if dropped["alignment"] else None,
            )
        else:
            if dropped["alignment"]:
                self.load_alignment_sidecar(str(dropped["alignment"][0]))
            if primary:
                self.load_subtitle_path(primary, primary=True)
            if secondary:
                self.load_subtitle_path(secondary, primary=False)
        event.acceptProposedAction()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.save_current_session()
        self.video_surface.shutdown()
        super().closeEvent(event)

    def handle_subtitle_interaction(self, interaction) -> None:
        if interaction.kind == "hover" and interaction.text == self._last_hover_text:
            return
        if interaction.kind == "hover":
            self._last_hover_text = interaction.text

        if interaction.kind == "selection":
            self.show_plugin_actions(interaction)
            return

        context = self.plugin_context_for(interaction)
        actions = self.plugin_registry.actions_for_event(interaction.kind)
        if not actions:
            return
        action = actions[0]
        result = self.plugin_registry.run(action.plugin_id, action.label, context)
        if interaction.kind == "hover":
            QToolTip.showText(QCursor.pos(), tooltip_text(result), self.overlay)
            return

        self.show_plugin_result(result)

    def show_plugin_actions(self, interaction: SubtitleInteraction) -> None:
        context = self.plugin_context_for(interaction)
        self.current_plugin_context = context
        lines = [
            f"Selected segment at {format_time(interaction.timestamp)}",
            "",
            f"Primary:\n{context.primary_text or '(none)'}",
            "",
            f"Secondary:\n{context.secondary_text or '(none)'}",
            "",
            "Available plugin actions:",
        ]
        for action in self.plugin_registry.actions_for_event("selection"):
            lines.append(f"- {action.label}")
        self.show_plugin_result(PluginResult("Language Tools", "\n".join(lines), "plugin-registry"))
        self.set_plugin_actions(context, "selection")

    def set_plugin_actions(self, context: PluginContext, event_kind: str) -> None:
        while self.plugin_actions_bar.count() > 1:
            item = self.plugin_actions_bar.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.plugin_action_buttons = []

        for action in self.plugin_registry.actions_for_event(event_kind):
            button = QPushButton(action.label)
            button.clicked.connect(
                lambda _checked=False, plugin_id=action.plugin_id, label=action.label: self.run_plugin_action(
                    plugin_id,
                    label,
                    context,
                    async_run=plugin_id in {"phrase_explanation", "sentence_diagram"},
                )
            )
            self.plugin_actions_bar.insertWidget(max(0, self.plugin_actions_bar.count() - 1), button)
            self.plugin_action_buttons.append(button)

    def run_plugin_action(self, plugin_id: str, action: str, context: PluginContext, *, async_run: bool) -> None:
        if not async_run:
            self.show_plugin_result(self.plugin_registry.run(plugin_id, action, context))
            return

        self.show_plugin_result(PluginResult(f"Running {action}", "Working...", f"plugin:{plugin_id}:pending"))
        worker = PluginWorker(self.plugin_registry, plugin_id, action, context)
        worker.signals.finished.connect(self.show_plugin_result)
        worker.signals.failed.connect(self.show_plugin_error)
        self.thread_pool.start(worker)

    def plugin_context_for(self, interaction: SubtitleInteraction) -> PluginContext:
        primary = self.overlay.snapshot.primary
        secondary = self.overlay.snapshot.secondary
        return PluginContext(
            interaction=interaction,
            primary_text=primary,
            secondary_text=secondary,
            timestamp=interaction.timestamp,
        )

    def show_plugin_result(self, result: PluginResult) -> None:
        self.plugin_dock.show()
        self.plugin_dock.setWindowTitle(result.title)
        if result.content_type == "html":
            self.plugin_output.setHtml(f"{result.body}<p style='color:#94a3b8;'>Source: {html_escape(result.source)}</p>")
        else:
            self.plugin_output.setPlainText(f"{result.title}\n\n{result.body}\n\nSource: {result.source}")

    def show_plugin_error(self, message: str) -> None:
        self.show_plugin_result(PluginResult("Language Tool Error", message, "plugin:error"))

    def open_llm_settings(self) -> None:
        dialog = LLMSettingsDialog(self.llm_settings, self)
        dialog.settings_saved.connect(self.apply_llm_settings)
        dialog.exec()

    def apply_llm_settings(self, settings: LLMEndpointSettings) -> None:
        self.llm_settings = settings
        save_llm_settings(settings)
        self.plugin_registry = create_default_registry(llm_settings=settings)
        self.alignment_panel.apply_llm_settings(settings)

    def toggle_script_sidebar(self) -> None:
        self.script_dock.setVisible(not self.script_dock.isVisible())

    def toggle_alignment_panel(self) -> None:
        self.alignment_panel.set_loaded_paths(
            self.primary_subtitle_path,
            self.secondary_subtitle_path,
            clear_missing=True,
        )
        self.alignment_dock.setVisible(not self.alignment_dock.isVisible())

    def run_alignment(self, config: dict) -> None:
        worker = AlignmentWorker(config)
        worker.signals.progress.connect(self.alignment_panel.append_log)
        worker.signals.finished.connect(self.alignment_panel.mark_finished)
        worker.signals.failed.connect(self.alignment_panel.mark_failed)
        self.thread_pool.start(worker)

    def load_alignment_sidecar(self, sidecar_path: str) -> None:
        try:
            package = load_alignment_package(sidecar_path)
            primary, secondary = tracks_from_alignment_package(package)
            self.select_embedded_track(primary=True, track_id=None, clear_external=False)
            self.select_embedded_track(primary=False, track_id=None, clear_external=False)
            self.subtitle_engine.primary = primary
            self.subtitle_engine.secondary = secondary
            self.primary_subtitle_path = package.primary_source
            self.secondary_subtitle_path = package.secondary_source
            self.alignment_sidecar_path = sidecar_path
            self.script_sidebar.set_tracks(self.subtitle_engine.primary, self.subtitle_engine.secondary)
            self.alignment_panel.set_loaded_paths(
                self.primary_subtitle_path,
                self.secondary_subtitle_path,
                clear_missing=True,
            )
        except Exception as exc:  # noqa: BLE001 - show parser/file failures to the user.
            QMessageBox.critical(self, "Alignment load error", str(exc))

    def seek_to(self, timestamp: float) -> None:
        player = self.video_surface.player
        if player is not None:
            player.seek(timestamp, reference="absolute")


def format_time(value: float | None) -> str:
    if value is None:
        value = 0
    total_seconds = max(0, int(value))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def tooltip_text(result: PluginResult) -> str:
    body = result.body.strip()
    if len(body) > 220:
        body = body[:217].rstrip() + "..."
    return f"{result.title}\n\n{body}"


def html_escape(value: str) -> str:
    return html.escape(value, quote=True)


def subtitle_offset_control(tooltip: str) -> QDoubleSpinBox:
    control = QDoubleSpinBox()
    control.setRange(-30.0, 30.0)
    control.setSingleStep(0.1)
    control.setDecimals(1)
    control.setValue(0.0)
    control.setSuffix("s")
    control.setToolTip(f"{tooltip}; positive values show subtitles later")
    return control


def safe_mpv_text(player: mpv.MPV, property_name: str) -> str:
    try:
        value = getattr(player, property_name)
    except Exception:  # noqa: BLE001 - properties can be unavailable during seeks/loading.
        return ""
    return value if isinstance(value, str) else ""


def preferred_embedded_track(
    tracks: list[dict],
    language: str,
    *,
    excluded: int | None,
) -> int | None:
    candidates = [
        track
        for track in tracks
        if isinstance(track.get("id"), int)
        and track.get("id") != excluded
        and is_text_subtitle_track(track)
    ]
    language_match = next(
        (
            track
            for track in candidates
            if language_matches(Path(f"video.{track.get('lang', '')}.srt"), language)
        ),
        None,
    )
    default_match = next((track for track in candidates if track.get("default")), None)
    selected = language_match or default_match or (candidates[0] if candidates else None)
    return int(selected["id"]) if selected is not None else None
