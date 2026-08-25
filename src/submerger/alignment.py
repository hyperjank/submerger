from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable
import urllib.error
import urllib.request

from .subtitles import SubtitleCue, SubtitleTrack, format_srt_timestamp, parse_srt


ALIGNMENT_SCHEMA_VERSION = 2
ALIGNMENT_PIPELINE_VERSION = "2026-08-25.1"


SENTENCE_END_RE = re.compile(r'[.!?。！？]["\'”’)\]]*$')
ELLIPSIS_END_RE = re.compile(r'(?:\.\.\.|…)["\'”’)\]]*$')


@dataclass(frozen=True)
class SubtitleDocument:
    language: str
    cues: list[SubtitleCue]
    path: str | None = None

    @classmethod
    def from_srt(cls, path: str | Path, language: str) -> "SubtitleDocument":
        return cls(language=language, cues=parse_srt(Path(path).read_text(encoding="utf-8-sig")), path=str(path))


@dataclass(frozen=True)
class PrimarySegment:
    segment_id: str
    start: float
    end: float
    text: str
    source_cue_ids: list[str]


@dataclass(frozen=True)
class CandidateWindow:
    primary: PrimarySegment
    secondary_cues: list[SubtitleCue]


@dataclass(frozen=True)
class AlignmentResult:
    primary_id: str
    secondary_cue_ids: list[str]
    confidence: float
    notes: str = ""


@dataclass(frozen=True)
class ValidatedAlignment:
    result: AlignmentResult
    status: str
    problems: list[str] = field(default_factory=list)
    accepted_secondary_cue_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PairedSegment:
    segment_id: str
    start: float
    end: float
    primary_text: str
    secondary_text: str
    primary_cue_ids: list[str]
    secondary_cue_ids: list[str]
    confidence: float
    status: str
    problems: list[str] = field(default_factory=list)
    candidate_secondary_cues: list[SubtitleCue] = field(default_factory=list)


@dataclass(frozen=True)
class AlignmentPackage:
    primary_language: str
    secondary_language: str
    segments: list[PairedSegment]
    issues: list[ValidatedAlignment]
    schema_version: int = ALIGNMENT_SCHEMA_VERSION
    primary_source: str | None = None
    secondary_source: str | None = None
    cache_identity: dict = field(default_factory=dict)


class PrimarySegmenter:
    def __init__(self, *, max_gap: float = 1.2, max_duration: float = 18.0, max_chars: int = 320) -> None:
        self.max_gap = max_gap
        self.max_duration = max_duration
        self.max_chars = max_chars

    def segment(self, document: SubtitleDocument) -> list[PrimarySegment]:
        segments: list[PrimarySegment] = []
        pending: list[SubtitleCue] = []

        for cue in document.cues:
            if pending and cue.start - pending[-1].end > self.max_gap:
                self._flush(segments, pending)
                pending = []

            pending.append(cue)
            text = normalize_segment_text(cue.text for cue in pending)
            duration = pending[-1].end - pending[0].start
            if has_terminal_sentence_end(text) or duration >= self.max_duration or len(text) >= self.max_chars:
                self._flush(segments, pending)
                pending = []

        if pending:
            self._flush(segments, pending)

        return segments

    def _flush(self, segments: list[PrimarySegment], cues: list[SubtitleCue]) -> None:
        segment_number = len(segments) + 1
        segments.append(
            PrimarySegment(
                segment_id=f"p_{segment_number:05d}",
                start=cues[0].start,
                end=cues[-1].end,
                text=normalize_segment_text(cue.text for cue in cues),
                source_cue_ids=[cue.cue_id for cue in cues],
            )
        )


class SecondaryWindowBuilder:
    def __init__(self, *, pad_seconds: float = 3.0, max_pad_seconds: float = 10.0) -> None:
        self.pad_seconds = pad_seconds
        self.max_pad_seconds = max_pad_seconds

    def build(self, primary_segments: list[PrimarySegment], secondary: SubtitleDocument) -> list[CandidateWindow]:
        return [CandidateWindow(primary=segment, secondary_cues=self._candidates(segment, secondary)) for segment in primary_segments]

    def _candidates(self, segment: PrimarySegment, secondary: SubtitleDocument) -> list[SubtitleCue]:
        pad = self.pad_seconds
        candidates: list[SubtitleCue] = []
        while pad <= self.max_pad_seconds:
            start = segment.start - pad
            end = segment.end + pad
            candidates = [cue for cue in secondary.cues if cue.end >= start and cue.start <= end]
            if candidates:
                return candidates
            pad *= 2
        return candidates


class HeuristicAlignmentClient:
    """Deterministic local aligner for development, tests, and cache warmups."""

    def cache_identity(self) -> dict:
        return {"provider": "heuristic", "version": 1}

    def align_batch(self, windows: list[CandidateWindow]) -> list[AlignmentResult]:
        results: list[AlignmentResult] = []
        for window in windows:
            overlapping = [
                cue.cue_id
                for cue in window.secondary_cues
                if cue.end >= window.primary.start and cue.start <= window.primary.end
            ]
            selected = overlapping or [cue.cue_id for cue in window.secondary_cues[:1]]
            confidence = 0.55 if selected else 0.0
            results.append(AlignmentResult(window.primary.segment_id, selected, confidence, "heuristic time overlap"))
        return results


class OpenAICompatibleAlignmentClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("SUBMERGER_LLM_MODEL") or "gpt-4.1-mini"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("SUBMERGER_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI-compatible alignment client.")

    def cache_identity(self) -> dict:
        return {
            "provider": "openai-compatible",
            "model": self.model,
            "base_url": self.base_url,
            "prompt_version": ALIGNMENT_PIPELINE_VERSION,
        }

    def align_batch(self, windows: list[CandidateWindow]) -> list[AlignmentResult]:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": alignment_response_format(),
            "messages": [
                {"role": "system", "content": ALIGNMENT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"windows": [window_payload(w) for w in windows]}, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM alignment request failed: {exc.code} {detail}") from exc

        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return [
            AlignmentResult(
                primary_id=item["primary_id"],
                secondary_cue_ids=[str(cue_id) for cue_id in item.get("secondary_cue_ids", [])],
                confidence=float(item.get("confidence", 0)),
                notes=str(item.get("notes", "")),
            )
            for item in parsed.get("alignments", [])
        ]


ALIGNMENT_SYSTEM_PROMPT = """You align bilingual subtitles for language learning.
Return only JSON with an "alignments" array.
For each primary segment, choose only secondary cue ids from the provided candidates that match the complete semantic content of the primary text.
Return cue ids, not rewritten text. Preserve monotonic order. Use an empty list and low confidence when no candidate is a good match.
Each item must include primary_id, secondary_cue_ids, confidence from 0 to 1, and notes."""


class AlignmentValidator:
    def __init__(self, *, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def validate(
        self,
        results: list[AlignmentResult],
        windows: list[CandidateWindow],
    ) -> list[ValidatedAlignment]:
        expected_ids = {window.primary.segment_id for window in windows}
        grouped: dict[str, list[AlignmentResult]] = {primary_id: [] for primary_id in expected_ids}
        unknown_results: list[AlignmentResult] = []
        for result in results:
            if result.primary_id in grouped:
                grouped[result.primary_id].append(result)
            else:
                unknown_results.append(result)

        seen: set[str] = set()
        last_start = -1.0
        validated: list[ValidatedAlignment] = []

        for window in windows:
            problems: list[str] = []
            entries = grouped[window.primary.segment_id]
            if not entries:
                result = AlignmentResult(window.primary.segment_id, [], 0.0, "missing model response")
                problems.append("missing alignment result")
            else:
                if len(entries) > 1:
                    problems.append(f"duplicate primary result entries: {len(entries)} (merged)")
                cue_ids = [cue_id for entry in entries for cue_id in entry.secondary_cue_ids]
                duplicate_ids = list(dict.fromkeys(
                    cue_id for cue_id in cue_ids if cue_ids.count(cue_id) > 1
                ))
                if duplicate_ids:
                    problems.append(f"duplicate secondary cue ids: {', '.join(duplicate_ids)}")
                unique_ids = list(dict.fromkeys(cue_ids))
                raw_confidence = min(entry.confidence for entry in entries)
                if not 0.0 <= raw_confidence <= 1.0:
                    problems.append("confidence outside 0..1")
                notes = "; ".join(dict.fromkeys(entry.notes for entry in entries if entry.notes))
                result = AlignmentResult(
                    window.primary.segment_id,
                    unique_ids,
                    min(1.0, max(0.0, raw_confidence)),
                    notes,
                )

            candidate_lookup = {cue.cue_id: cue for cue in window.secondary_cues}
            outside_window = [
                cue_id for cue_id in result.secondary_cue_ids if cue_id not in candidate_lookup
            ]
            reused = [cue_id for cue_id in result.secondary_cue_ids if cue_id in seen]
            selected = [
                candidate_lookup[cue_id]
                for cue_id in result.secondary_cue_ids
                if cue_id in candidate_lookup
            ]

            if outside_window:
                problems.append(f"secondary cue ids outside candidate window: {', '.join(outside_window)}")
            if reused:
                problems.append(f"reused secondary cue ids: {', '.join(reused)}")
            if result.confidence < self.min_confidence:
                problems.append("low confidence")
            if selected and selected[0].start < last_start:
                problems.append("non-monotonic secondary order")
            if any(left.start > right.start for left, right in zip(selected, selected[1:])):
                problems.append("non-monotonic secondary order")
            if not selected:
                problems.append("no secondary match")

            accepted_ids: list[str] = []
            if result.confidence >= self.min_confidence:
                available = sorted(
                    (
                        cue
                        for cue in selected
                        if cue.cue_id not in seen and cue.start >= last_start
                    ),
                    key=lambda cue: cue.start,
                )
                accepted_ids = [cue.cue_id for cue in available]

            if not problems:
                status = "ok"
            elif accepted_ids:
                status = "repaired"
            else:
                status = "needs_review"
            validated.append(ValidatedAlignment(result=result, status=status, problems=problems, accepted_secondary_cue_ids=accepted_ids))

            for cue_id in accepted_ids:
                seen.add(cue_id)
            if accepted_ids:
                last_start = candidate_lookup[accepted_ids[-1]].start

        for result in unknown_results:
            validated.append(
                ValidatedAlignment(
                    result=result,
                    status="needs_review",
                    problems=[f"unknown primary id: {result.primary_id}"],
                    accepted_secondary_cue_ids=[],
                )
            )

        return validated


def build_alignment_identity(
    *,
    primary_path: str | Path,
    secondary_path: str | Path,
    primary_language: str,
    secondary_language: str,
    client,
    batch_size: int,
    pad_seconds: float,
    drop_non_dialogue: bool,
) -> dict:
    client_identity = (
        client.cache_identity()
        if callable(getattr(client, "cache_identity", None))
        else {"provider": type(client).__name__}
    )
    return {
        "pipeline_version": ALIGNMENT_PIPELINE_VERSION,
        "primary_sha256": file_sha256(primary_path),
        "secondary_sha256": file_sha256(secondary_path),
        "primary_language": primary_language,
        "secondary_language": secondary_language,
        "batch_size": batch_size,
        "pad_seconds": pad_seconds,
        "drop_non_dialogue": drop_non_dialogue,
        "client": client_identity,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_subtitles(
    primary_path: str | Path,
    secondary_path: str | Path,
    *,
    primary_language: str = "primary",
    secondary_language: str = "secondary",
    client: HeuristicAlignmentClient | OpenAICompatibleAlignmentClient | None = None,
    batch_size: int = 12,
    pad_seconds: float = 3.0,
    progress: Callable[[int, int], None] | None = None,
    drop_non_dialogue: bool = True,
    cache_path: str | Path | None = None,
) -> AlignmentPackage:
    primary = SubtitleDocument.from_srt(primary_path, primary_language)
    secondary = SubtitleDocument.from_srt(secondary_path, secondary_language)
    if drop_non_dialogue:
        primary = filter_dialogue_document(primary)
        secondary = filter_dialogue_document(secondary)
    segments = PrimarySegmenter().segment(primary)
    windows = SecondaryWindowBuilder(pad_seconds=pad_seconds).build(segments, secondary)
    client = client or HeuristicAlignmentClient()

    cache_identity = build_alignment_identity(
        primary_path=primary_path,
        secondary_path=secondary_path,
        primary_language=primary_language,
        secondary_language=secondary_language,
        client=client,
        batch_size=batch_size,
        pad_seconds=pad_seconds,
        drop_non_dialogue=drop_non_dialogue,
    )

    cached_batches = load_alignment_cache(cache_path, expected_identity=cache_identity)
    raw_results: list[AlignmentResult] = []
    total_batches = max(1, (len(windows) + batch_size - 1) // batch_size)
    for batch_number, index in enumerate(range(0, len(windows), batch_size), start=1):
        if progress is not None:
            progress(batch_number, total_batches)
        cache_key = str(index)
        if cache_key in cached_batches:
            raw_results.extend(cached_batches[cache_key])
            continue
        batch_results = client.align_batch(windows[index : index + batch_size])
        cached_batches[cache_key] = batch_results
        save_alignment_cache(cache_path, cached_batches, identity=cache_identity)
        raw_results.extend(batch_results)

    validated = AlignmentValidator().validate(raw_results, windows)
    result_lookup = {item.result.primary_id: item for item in validated}
    paired = [
        make_paired_segment(window, result_lookup.get(window.primary.segment_id))
        for window in windows
    ]
    issues = [item for item in validated if item.status != "ok"]
    return AlignmentPackage(
        primary_language=primary_language,
        secondary_language=secondary_language,
        segments=paired,
        issues=issues,
        primary_source=str(Path(primary_path).resolve()),
        secondary_source=str(Path(secondary_path).resolve()),
        cache_identity=cache_identity,
    )


def make_paired_segment(
    window: CandidateWindow,
    validated: ValidatedAlignment | None,
) -> PairedSegment:
    segment = window.primary
    if validated is None:
        return PairedSegment(
            segment.segment_id,
            segment.start,
            segment.end,
            segment.text,
            "",
            segment.source_cue_ids,
            [],
            0,
            "needs_review",
            ["missing alignment result"],
            list(window.secondary_cues),
        )

    secondary_lookup = {cue.cue_id: cue for cue in window.secondary_cues}
    secondary_cues = [
        secondary_lookup[cue_id]
        for cue_id in validated.accepted_secondary_cue_ids
        if cue_id in secondary_lookup
    ]
    return PairedSegment(
        segment_id=segment.segment_id,
        start=segment.start,
        end=segment.end,
        primary_text=segment.text,
        secondary_text=normalize_segment_text(cue.text for cue in secondary_cues),
        primary_cue_ids=segment.source_cue_ids,
        secondary_cue_ids=validated.accepted_secondary_cue_ids,
        confidence=validated.result.confidence,
        status=validated.status,
        problems=validated.problems,
        candidate_secondary_cues=list(window.secondary_cues),
    )


def write_alignment_outputs(package: AlignmentPackage, output_prefix: str | Path) -> tuple[Path, Path, Path]:
    existing_path = alignment_artifact_path(output_prefix, ".alignment.json")
    if existing_path.exists():
        try:
            package = carry_forward_human_reviews(
                package,
                load_alignment_package(existing_path),
            )
        except (OSError, ValueError):
            pass
    json_path = write_alignment_sidecar(package, output_prefix)
    primary_path, secondary_path = write_aligned_srt_exports(package, output_prefix)
    return primary_path, secondary_path, json_path


def write_alignment_sidecar(package: AlignmentPackage, output_prefix: str | Path) -> Path:
    path = alignment_artifact_path(output_prefix, ".alignment.json")
    atomic_write_text(path, json.dumps(asdict(package), ensure_ascii=False, indent=2))
    return path


def write_aligned_srt_exports(
    package: AlignmentPackage,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    primary_path = alignment_artifact_path(
        output_prefix,
        f".{package.primary_language}.clean.srt",
    )
    secondary_path = alignment_artifact_path(
        output_prefix,
        f".{package.secondary_language}.aligned.srt",
    )
    atomic_write_text(
        primary_path,
        render_srt([(s.start, s.end, s.primary_text) for s in package.segments]),
    )
    atomic_write_text(
        secondary_path,
        render_srt([(s.start, s.end, s.secondary_text) for s in package.segments]),
    )
    return primary_path, secondary_path


def alignment_artifact_path(output_prefix: str | Path, suffix: str) -> Path:
    prefix = Path(output_prefix)
    if str(prefix).endswith(suffix):
        return prefix
    return Path(f"{prefix}{suffix}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def load_alignment_package(path: str | Path) -> AlignmentPackage:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Alignment sidecar must contain a JSON object.")
    segments = [paired_segment_from_dict(item) for item in data.get("segments", [])]
    issues = [validated_alignment_from_dict(item) for item in data.get("issues", [])]
    return AlignmentPackage(
        primary_language=str(data.get("primary_language", "primary")),
        secondary_language=str(data.get("secondary_language", "secondary")),
        segments=segments,
        issues=issues,
        schema_version=int(data.get("schema_version", 1)),
        primary_source=data.get("primary_source"),
        secondary_source=data.get("secondary_source"),
        cache_identity=data.get("cache_identity", {}) if isinstance(data.get("cache_identity", {}), dict) else {},
    )


def paired_segment_from_dict(data: dict) -> PairedSegment:
    return PairedSegment(
        segment_id=str(data.get("segment_id", "")),
        start=float(data.get("start", 0)),
        end=float(data.get("end", 0)),
        primary_text=str(data.get("primary_text", "")),
        secondary_text=str(data.get("secondary_text", "")),
        primary_cue_ids=[str(value) for value in data.get("primary_cue_ids", [])],
        secondary_cue_ids=[str(value) for value in data.get("secondary_cue_ids", [])],
        confidence=float(data.get("confidence", 0)),
        status=str(data.get("status", "needs_review")),
        problems=[str(value) for value in data.get("problems", [])],
        candidate_secondary_cues=[
            subtitle_cue_from_dict(value)
            for value in data.get("candidate_secondary_cues", [])
            if isinstance(value, dict)
        ],
    )


def subtitle_cue_from_dict(data: dict) -> SubtitleCue:
    return SubtitleCue(
        start=float(data.get("start", 0)),
        end=float(data.get("end", 0)),
        text=str(data.get("text", "")),
        cue_id=str(data.get("cue_id", "")),
        raw_text=str(data.get("raw_text", "")),
    )


def validated_alignment_from_dict(data: dict) -> ValidatedAlignment:
    result_data = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
    result = AlignmentResult(
        primary_id=str(result_data.get("primary_id", "")),
        secondary_cue_ids=[str(value) for value in result_data.get("secondary_cue_ids", [])],
        confidence=float(result_data.get("confidence", 0)),
        notes=str(result_data.get("notes", "")),
    )
    return ValidatedAlignment(
        result=result,
        status=str(data.get("status", "needs_review")),
        problems=[str(value) for value in data.get("problems", [])],
        accepted_secondary_cue_ids=[
            str(value) for value in data.get("accepted_secondary_cue_ids", [])
        ],
    )


def tracks_from_alignment_package(
    package: AlignmentPackage,
) -> tuple[SubtitleTrack, SubtitleTrack]:
    primary = SubtitleTrack([
        SubtitleCue(
            segment.start,
            segment.end,
            segment.primary_text,
            segment.segment_id,
            segment.primary_text,
        )
        for segment in package.segments
        if segment.primary_text.strip()
    ])
    secondary = SubtitleTrack([
        SubtitleCue(
            segment.start,
            segment.end,
            segment.secondary_text,
            segment.segment_id,
            segment.secondary_text,
        )
        for segment in package.segments
        if segment.secondary_text.strip()
    ])
    return primary, secondary


def review_alignment_segment(
    package: AlignmentPackage,
    segment_id: str,
    selected_secondary_cue_ids: list[str] | None = None,
) -> AlignmentPackage:
    segments = list(package.segments)
    for index, segment in enumerate(segments):
        if segment.segment_id != segment_id:
            continue
        candidate_lookup = {cue.cue_id: cue for cue in segment.candidate_secondary_cues}
        requested_ids = (
            segment.secondary_cue_ids
            if selected_secondary_cue_ids is None
            else selected_secondary_cue_ids
        )
        if not candidate_lookup and requested_ids == segment.secondary_cue_ids:
            segments[index] = replace(
                segment,
                confidence=1.0,
                status="reviewed",
                problems=[],
            )
            issues = [
                issue for issue in package.issues if issue.result.primary_id != segment_id
            ]
            return replace(package, segments=segments, issues=issues)
        unknown = [cue_id for cue_id in requested_ids if cue_id not in candidate_lookup]
        if unknown:
            raise ValueError(f"Cue ids are not review candidates: {', '.join(unknown)}")
        selected = sorted(
            (candidate_lookup[cue_id] for cue_id in dict.fromkeys(requested_ids)),
            key=lambda cue: cue.start,
        )
        segments[index] = replace(
            segment,
            secondary_text=normalize_segment_text(cue.text for cue in selected),
            secondary_cue_ids=[cue.cue_id for cue in selected],
            confidence=1.0,
            status="reviewed",
            problems=[],
        )
        issues = [
            issue for issue in package.issues if issue.result.primary_id != segment_id
        ]
        return replace(package, segments=segments, issues=issues)
    raise KeyError(f"Unknown alignment segment: {segment_id}")


def carry_forward_human_reviews(
    package: AlignmentPackage,
    previous: AlignmentPackage,
) -> AlignmentPackage:
    identity_keys = ("primary_sha256", "secondary_sha256")
    if any(
        not package.cache_identity.get(key)
        or package.cache_identity.get(key) != previous.cache_identity.get(key)
        for key in identity_keys
    ):
        return package

    reviewed = {
        segment.segment_id: segment
        for segment in previous.segments
        if segment.status == "reviewed"
    }
    if not reviewed:
        return package

    retained_ids: set[str] = set()
    segments: list[PairedSegment] = []
    for segment in package.segments:
        old = reviewed.get(segment.segment_id)
        if old is None or old.primary_text != segment.primary_text:
            segments.append(segment)
            continue
        candidate_ids = {cue.cue_id for cue in segment.candidate_secondary_cues}
        if any(cue_id not in candidate_ids for cue_id in old.secondary_cue_ids):
            segments.append(segment)
            continue
        segments.append(
            replace(
                segment,
                secondary_text=old.secondary_text,
                secondary_cue_ids=list(old.secondary_cue_ids),
                confidence=1.0,
                status="reviewed",
                problems=[],
            )
        )
        retained_ids.add(segment.segment_id)

    return replace(
        package,
        segments=segments,
        issues=[
            issue
            for issue in package.issues
            if issue.result.primary_id not in retained_ids
        ],
    )


def render_srt(rows: list[tuple[float, float, str]]) -> str:
    blocks: list[str] = []
    for index, (start, end, text) in enumerate(rows, start=1):
        blocks.append(f"{index}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text.strip()}")
    return "\n\n".join(blocks) + "\n"


def normalize_segment_text(lines) -> str:
    text = " ".join(line.replace("\n", " ").strip() for line in lines if line.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:。！？；：])", r"\1", text)
    return text.strip()


def has_terminal_sentence_end(text: str) -> bool:
    stripped = text.strip()
    return bool(SENTENCE_END_RE.search(stripped)) and not ELLIPSIS_END_RE.search(stripped)


def filter_dialogue_document(document: SubtitleDocument) -> SubtitleDocument:
    return SubtitleDocument(
        language=document.language,
        cues=[cue for cue in document.cues if not is_non_dialogue_cue(cue)],
        path=document.path,
    )


def is_non_dialogue_cue(cue: SubtitleCue) -> bool:
    text = cue.text.strip()
    raw = cue.raw_text.strip()
    if not text:
        return True
    if "{\\an" in raw:
        return True
    lowered = text.lower()
    if "opensubtitles" in lowered or "advertise your product" in lowered:
        return True
    if text.startswith("【") and text.endswith("】"):
        return True
    without_brackets = re.sub(r"\[[^\]]+\]", "", text).strip()
    without_music = without_brackets.replace("♪", "").strip()
    return not re.search(r"[A-Za-z0-9\u3400-\u9fff]", without_music)


def window_payload(window: CandidateWindow) -> dict:
    return {
        "primary": asdict(window.primary),
        "secondary_candidates": [
            {"cue_id": cue.cue_id, "start": cue.start, "end": cue.end, "text": cue.text}
            for cue in window.secondary_cues
        ],
    }


def load_alignment_cache(
    cache_path: str | Path | None,
    *,
    expected_identity: dict | None = None,
) -> dict[str, list[AlignmentResult]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if expected_identity is not None and data.get("identity") != expected_identity:
        return {}
    return {
        key: [
            AlignmentResult(
                primary_id=item["primary_id"],
                secondary_cue_ids=[str(cue_id) for cue_id in item.get("secondary_cue_ids", [])],
                confidence=float(item.get("confidence", 0)),
                notes=str(item.get("notes", "")),
            )
            for item in value
        ]
        for key, value in data.get("batches", {}).items()
    }


def save_alignment_cache(
    cache_path: str | Path | None,
    batches: dict[str, list[AlignmentResult]],
    *,
    identity: dict | None = None,
) -> None:
    if cache_path is None:
        return
    path = Path(cache_path)
    atomic_write_text(
        path,
        json.dumps(
            {
                "version": 2,
                "identity": identity or {},
                "batches": {
                    key: [asdict(item) for item in value]
                    for key, value in batches.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def alignment_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "subtitle_alignment_response",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "alignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "primary_id": {"type": "string"},
                                "secondary_cue_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "confidence": {"type": "number"},
                                "notes": {"type": "string"},
                            },
                            "required": ["primary_id", "secondary_cue_ids", "confidence", "notes"],
                        },
                    }
                },
                "required": ["alignments"],
            },
        },
    }
