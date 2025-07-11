#!/usr/bin/env python3
"""
LLM-assisted subtitle alignment (align.py)
Integrates:
 - regex-based junk stripping + LLM cleanup of first/last 30s
 - utterance merging for SL
 - block-based TL selection via small local LLM
"""
import os
import re
import json
from pathlib import Path
from typing import List
from openai import OpenAI
from . import sync

# Patterns
JUNK_RE = re.compile(r"(https?://\S+|www\.\S+|presented by|translator)", re.IGNORECASE)
SENT_END_RE = re.compile(r"[\.\!\?。！？…]+['\"]?$")
TS_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})")


# LLM client loader
def get_client() -> tuple[OpenAI, str]:
    """Return an ``OpenAI`` client and the default model name."""
    cfg_path = Path(__file__).with_name("settings.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    llm = cfg.get("llm", {})
    key = llm.get("api_key") or os.getenv("OPENAI_API_KEY")
    base = llm.get("api_base") or os.getenv(
        "OPENAI_API_BASE", "http://localhost:1234/v1"
    )
    model = llm.get("model") or os.getenv("OPENAI_MODEL", "qwen3-8b")
    if not key:
        raise RuntimeError(
            "Please configure llm.api_key in settings.json or set OPENAI_API_KEY"
        )

    client = OpenAI(api_key=key, base_url=base)
    return client, model


# Helpers to parse SRT timestamps


def ts_to_ms(ts: str) -> int:
    m = TS_RE.match(ts)
    if not m:
        return 0
    h, mi, s, ms = m.group("h", "m", "s", "ms")
    return (int(h) * 3600 + int(mi) * 60 + int(s)) * 1000 + int(ms)


# Sample first/last window of cues
def sample_segments(
    cues: List[sync.SubtitleCue], window_ms: int = 30000
) -> List[sync.SubtitleCue]:
    if not cues:
        return []
    total = cues[-1].end_time
    return [
        c for c in cues if c.start_time < window_ms or c.end_time > total - window_ms
    ]


# 1) Strip junk via regex
def strip_junk(
    cues: List[sync.SubtitleCue], window_ms: int = 30000
) -> List[sync.SubtitleCue]:
    if not cues:
        return []
    total = cues[-1].end_time
    out = []
    for c in cues:
        txt = c.text
        if c.start_time < window_ms or c.end_time > total - window_ms:
            txt = JUNK_RE.sub("", txt).strip()
        if txt:
            out.append(sync.SubtitleCue(c.start_time, c.end_time, txt))
    return out


# 2) LLM cleanup of sample


def call_llm_cleanup(
    tl_sample: List[sync.SubtitleCue], sl_sample: List[sync.SubtitleCue]
) -> (List[sync.SubtitleCue], List[sync.SubtitleCue]):
    client, model = get_client()

    # build SRT-like blocks
    def block(cues, tag):
        lines = [f"{tag}"]
        for c in cues:
            # format hh:mm:ss,ms --> hh:mm:ss,ms text
            start = f"{c.start_time//3600000:02}:{(c.start_time//60000)%60:02}:{(c.start_time//1000)%60:02},{c.start_time%1000:03}"
            end = f"{c.end_time//3600000:02}:{(c.end_time//60000)%60:02}:{(c.end_time//1000)%60:02},{c.end_time%1000:03}"
            lines.append(f"{start} --> {end} {c.text}")
        return "\n".join(lines)

    prompt = (
        "You are given two subtitle samples. Remove lines that are ads, group credits, or other non-dialog."
        "Return cleaned samples in the same format, preserving tags."
        f"\n\n{block(tl_sample,'===TL===')}\n\n{block(sl_sample,'===SL===')}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500,
    )
    text = resp.choices[0].message.content
    # split back
    tl_cleaned, sl_cleaned = [], []
    section = None
    for line in text.splitlines():
        if line.strip() == "===TL===":
            section = "tl"
            continue
        if line.strip() == "===SL===":
            section = "sl"
            continue
        if not section or "-->" not in line:
            continue
        ts, rest = line.split("-->", 1)
        start_ts = ts.strip()
        end_and_txt = rest.strip().split(" ", 1)
        if len(end_and_txt) < 2:
            continue
        end_ts, txt = end_and_txt
        cue = sync.SubtitleCue(ts_to_ms(start_ts), ts_to_ms(end_ts), txt.strip())
        (tl_cleaned if section == "tl" else sl_cleaned).append(cue)
    return tl_cleaned, sl_cleaned


# 3) Merge SL into utterances
def merge_utterances(
    cues: List[sync.SubtitleCue], gap_ms: int = 500
) -> List[sync.SubtitleCue]:
    merged, buf = [], []
    start = end = 0
    for c in cues:
        if not buf:
            start, end = c.start_time, c.end_time
        if buf and c.start_time - end > gap_ms:
            merged.append(sync.SubtitleCue(start, end, " ".join(buf)))
            buf = []
            start = c.start_time
        buf.append(c.text)
        end = c.end_time
        if SENT_END_RE.search(c.text):
            merged.append(sync.SubtitleCue(start, end, " ".join(buf)))
            buf = []
    if buf:
        merged.append(sync.SubtitleCue(start, end, " ".join(buf)))
    return merged


# 4) Call LLM to select matching TL text block


def call_llm_select_target(sl_text: str, tl_block: List[sync.SubtitleCue]) -> str:
    client, model = get_client()
    block_text = "\n".join(c.text for c in tl_block)
    prompt = (
        f"Source utterance: {sl_text}\n\n"
        f"Here is a block of target subtitles (text only):\n{block_text}\n\n"
        "Please return only the part of the above target text that best matches the source utterance."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


# Main entry point


def main(
    tl_file: str,
    sl_file: str,
    *,
    tl_code: str = "tl",
    sl_code: str = "sl",
    out: str = "final_synced",
) -> None:
    # load + dedupe
    tl = sync.dedupe_cues(sync.make_cued(tl_file))
    sl = sync.dedupe_cues(sync.make_cued(sl_file))

    # regex strip
    tl = strip_junk(tl)
    sl = strip_junk(sl)

    # LLM cleanup on first/last 30s
    tl_sample = sample_segments(tl)
    sl_sample = sample_segments(sl)
    tl_cl, sl_cl = call_llm_cleanup(tl_sample, sl_sample)
    # replace sample portions in full lists
    tl = [
        c for c in tl if c.start_time not in {s.start_time for s in tl_sample}
    ] + tl_cl
    sl = [
        c for c in sl if c.start_time not in {s.start_time for s in sl_sample}
    ] + sl_cl
    tl.sort(key=lambda c: c.start_time)
    sl.sort(key=lambda c: c.start_time)

    # utterance merge
    sl_utts = merge_utterances(sl)

    # build aligned list
    aligned = []
    for utt in sl_utts:
        # find overlapping TL indices
        overlaps = [
            i
            for i, c in enumerate(tl)
            if not (c.end_time < utt.start_time or c.start_time > utt.end_time)
        ]
        if overlaps:
            lo, hi = max(min(overlaps) - 2, 0), min(max(overlaps) + 3, len(tl))
            tl_block = tl[lo:hi]
        else:
            tl_block = []
        sel = call_llm_select_target(utt.text, tl_block)
        aligned.append(
            {
                "start_time": utt.start_time,
                "end_time": utt.end_time,
                "tl_text": sel,
                "sl_text": utt.text,
            }
        )

    # write out
    sync.write_synced_subs(aligned, out, tl_code, sl_code)
    print(f"Done! Wrote {out}_{tl_code}.srt and {out}_{sl_code}.srt")


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
