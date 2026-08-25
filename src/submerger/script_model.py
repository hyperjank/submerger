from __future__ import annotations

from dataclasses import dataclass

from .subtitles import SubtitleTrack, format_srt_timestamp


@dataclass(frozen=True)
class ScriptRow:
    start: float
    end: float
    primary: str
    secondary: str = ""

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end


def build_script_rows(primary: SubtitleTrack, secondary: SubtitleTrack) -> list[ScriptRow]:
    if primary.cues:
        return [
            ScriptRow(
                start=cue.start,
                end=cue.end,
                primary=cue.text,
                secondary=secondary.active_text((cue.start + cue.end) / 2),
            )
            for cue in primary.cues
        ]
    return [ScriptRow(cue.start, cue.end, "", cue.text) for cue in secondary.cues]


def render_script_row(row: ScriptRow) -> str:
    timestamp = f"{format_srt_timestamp(row.start)} - {format_srt_timestamp(row.end)}"
    lines = [timestamp]
    if row.primary:
        lines.append(row.primary)
    if row.secondary:
        lines.append(row.secondary)
    return "\n".join(lines)
