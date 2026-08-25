from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import html
import re
from pathlib import Path


TIMING_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
TAG_RE = re.compile(r"</?(?:font|b|i|u|s)\b[^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str
    cue_id: str = ""
    raw_text: str = ""

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end


class SubtitleTrack:
    def __init__(self, cues: list[SubtitleCue] | None = None) -> None:
        self.cues = sorted(cues or [], key=lambda cue: cue.start)
        self._starts = [cue.start for cue in self.cues]

    @classmethod
    def from_srt(cls, path: str | Path) -> "SubtitleTrack":
        text = Path(path).read_text(encoding="utf-8-sig")
        return cls(parse_srt(text))

    def active_text(self, timestamp: float | None) -> str:
        cue = self.active_cue(timestamp)
        return cue.text if cue is not None else ""

    def active_cue(self, timestamp: float | None) -> SubtitleCue | None:
        if timestamp is None or not self.cues:
            return None

        index = bisect_right(self._starts, timestamp) - 1
        if index < 0:
            return None

        cue = self.cues[index]
        return cue if cue.contains(timestamp) else None


class DualSubtitleEngine:
    def __init__(self) -> None:
        self.primary = SubtitleTrack()
        self.secondary = SubtitleTrack()

    def load_primary(self, path: str | Path) -> None:
        self.primary = SubtitleTrack.from_srt(path)

    def load_secondary(self, path: str | Path) -> None:
        self.secondary = SubtitleTrack.from_srt(path)

    def active(self, timestamp: float | None) -> tuple[str, str]:
        return (
            self.primary.active_text(timestamp),
            self.secondary.active_text(timestamp),
        )


def parse_srt(content: str) -> list[SubtitleCue]:
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n").strip())
    cues: list[SubtitleCue] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines()]
        if not lines:
            continue

        timing_index = next((i for i, line in enumerate(lines) if TIMING_RE.search(line)), None)
        if timing_index is None:
            continue

        match = TIMING_RE.search(lines[timing_index])
        if match is None:
            continue

        text_lines = lines[timing_index + 1 :]
        raw_text = "\n".join(text_lines)
        text = "\n".join(clean_subtitle_line(line) for line in text_lines).strip()
        if not text:
            continue

        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        if end <= start:
            continue

        cue_id = lines[0] if timing_index > 0 and lines[0] else str(len(cues) + 1)
        cues.append(SubtitleCue(start=start, end=end, text=text, cue_id=cue_id, raw_text=raw_text))

    return cues


def parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_subtitle_line(value: str) -> str:
    cleaned = re.sub(r"\{\\[^}]*}", "", value)
    cleaned = TAG_RE.sub("", cleaned)
    return html.unescape(cleaned).strip()


def format_srt_timestamp(value: float) -> str:
    total_ms = max(0, int(round(value * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
