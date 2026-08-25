from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
SUBTITLE_EXTENSIONS = {".srt"}
ALIGNMENT_SUFFIX = ".alignment.json"
STATE_VERSION = 1
IMAGE_SUBTITLE_CODECS = {
    "dvb_subtitle",
    "dvd_subtitle",
    "dvdsub",
    "hdmv_pgs_subtitle",
    "pgssub",
    "xsub",
}

LANGUAGE_ALIASES: dict[str, set[str]] = {
    "en": {"en", "eng", "english"},
    "zh": {"zh", "zho", "chi", "cmn", "chs", "cht", "chinese"},
    "es": {"es", "spa", "spanish"},
    "fr": {"fr", "fra", "fre", "french"},
    "de": {"de", "deu", "ger", "german"},
    "it": {"it", "ita", "italian"},
    "ja": {"ja", "jpn", "japanese"},
    "ko": {"ko", "kor", "korean"},
    "pt": {"pt", "por", "portuguese"},
    "ru": {"ru", "rus", "russian"},
}


@dataclass(frozen=True)
class PlaybackSession:
    video_path: str
    primary_subtitle_path: str | None = None
    secondary_subtitle_path: str | None = None
    alignment_sidecar_path: str | None = None
    position: float = 0.0
    speed: float = 1.0
    primary_offset: float = 0.0
    secondary_offset: float = 0.0
    primary_embedded_id: int | None = None
    secondary_embedded_id: int | None = None

    @property
    def title(self) -> str:
        return Path(self.video_path).name


class PlaybackSessionStore:
    def __init__(self, path: str | Path | None = None, *, limit: int = 10) -> None:
        self.path = Path(path).expanduser() if path is not None else default_playback_state_path()
        self.limit = limit

    def sessions(self) -> list[PlaybackSession]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
            return []
        raw_sessions = data.get("sessions")
        if not isinstance(raw_sessions, list):
            return []
        sessions: list[PlaybackSession] = []
        for item in raw_sessions:
            try:
                session = playback_session_from_dict(item)
            except (TypeError, ValueError):
                continue
            if Path(session.video_path).is_file():
                sessions.append(session)
        return sessions

    def last_session(self) -> PlaybackSession | None:
        sessions = self.sessions()
        return sessions[0] if sessions else None

    def remember(self, session: PlaybackSession) -> None:
        video_path = str(Path(session.video_path).expanduser().resolve())
        normalized = PlaybackSession(**{**asdict(session), "video_path": video_path})
        sessions = [
            item
            for item in self.sessions()
            if Path(item.video_path) != Path(video_path)
        ]
        sessions.insert(0, normalized)
        payload = {
            "version": STATE_VERSION,
            "sessions": [asdict(item) for item in sessions[: self.limit]],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def default_playback_state_path() -> Path:
    configured = os.environ.get("SUBMERGER_PLAYBACK_STATE_PATH")
    if configured:
        return Path(configured).expanduser()
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")).expanduser()
    return state_home / "submerger" / "playback.json"


def playback_session_from_dict(value: object) -> PlaybackSession:
    if not isinstance(value, dict) or not isinstance(value.get("video_path"), str):
        raise ValueError("invalid playback session")
    return PlaybackSession(
        video_path=value["video_path"],
        primary_subtitle_path=optional_string(value.get("primary_subtitle_path")),
        secondary_subtitle_path=optional_string(value.get("secondary_subtitle_path")),
        alignment_sidecar_path=optional_string(value.get("alignment_sidecar_path")),
        position=max(0.0, float(value.get("position", 0.0))),
        speed=min(3.0, max(0.25, float(value.get("speed", 1.0)))),
        primary_offset=min(30.0, max(-30.0, float(value.get("primary_offset", 0.0)))),
        secondary_offset=min(30.0, max(-30.0, float(value.get("secondary_offset", 0.0)))),
        primary_embedded_id=optional_int(value.get("primary_embedded_id")),
        secondary_embedded_id=optional_int(value.get("secondary_embedded_id")),
    )


def discover_external_subtitles(
    video_path: str | Path,
    *,
    primary_language: str = "en",
    secondary_language: str = "zh",
) -> tuple[Path | None, Path | None]:
    video = Path(video_path).expanduser().resolve()
    candidates = sorted(
        (
            path
            for path in video.parent.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUBTITLE_EXTENSIONS
            and subtitle_matches_video(path, video)
        ),
        key=lambda path: path.name.casefold(),
    )
    primary = first_language_match(candidates, primary_language)
    secondary = first_language_match(
        [path for path in candidates if path != primary],
        secondary_language,
    )
    remaining = [path for path in candidates if path not in {primary, secondary}]
    if primary is None and remaining:
        primary = remaining.pop(0)
    if secondary is None and remaining:
        secondary = remaining.pop(0)
    return primary, secondary


def discover_alignment_sidecar(video_path: str | Path) -> Path | None:
    video = Path(video_path).expanduser().resolve()
    exact = video.with_name(f"{video.stem}{ALIGNMENT_SUFFIX}")
    if exact.is_file():
        return exact
    candidates = sorted(video.parent.glob(f"{video.stem}*{ALIGNMENT_SUFFIX}"))
    return candidates[0] if candidates else None


def classify_dropped_paths(paths: list[str | Path]) -> dict[str, list[Path]]:
    classified: dict[str, list[Path]] = {"video": [], "subtitle": [], "alignment": []}
    for value in paths:
        path = Path(value).expanduser().resolve()
        lowered = path.name.lower()
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            classified["video"].append(path)
        elif path.suffix.lower() in SUBTITLE_EXTENSIONS:
            classified["subtitle"].append(path)
        elif lowered.endswith(ALIGNMENT_SUFFIX):
            classified["alignment"].append(path)
    return classified


def subtitle_matches_video(subtitle_path: Path, video_path: Path) -> bool:
    subtitle_stem = subtitle_path.stem.casefold()
    video_stem = video_path.stem.casefold()
    return subtitle_stem == video_stem or subtitle_stem.startswith(f"{video_stem}.")


def language_tokens(path: Path) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", path.stem.casefold())
        if token
    }


def language_matches(path: Path, language: str) -> bool:
    normalized = language.casefold().strip()
    aliases = LANGUAGE_ALIASES.get(normalized, {normalized})
    return bool(language_tokens(path) & aliases)


def first_language_match(candidates: list[Path], language: str) -> Path | None:
    return next((path for path in candidates if language_matches(path, language)), None)


def subtitle_track_label(track: dict) -> str:
    track_id = track.get("id", "?")
    language = track.get("lang") or "und"
    title = track.get("title")
    codec = track.get("codec")
    details = [str(language)]
    if title:
        details.append(str(title))
    if codec:
        details.append(str(codec))
    if track.get("default"):
        details.append("default")
    return f"{track_id}: {' · '.join(details)}"


def is_text_subtitle_track(track: dict) -> bool:
    return str(track.get("codec") or "").casefold() not in IMAGE_SUBTITLE_CODECS


def optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
