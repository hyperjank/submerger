#!/usr/bin/env python3
"""
Subtitle alignment pipeline:
1. Load and parse SRT files into cues
2. Strip garbage text from start/end
3. Merge into full utterances based on sentence boundaries
4. Extend timestamps to cover utterance duration
5. Align target language cues to source utterances via LLM
6. Write out synchronized SRTs
"""
import re
import json
from pathlib import Path
from typing import List, Tuple
import pysubs2
from openai import OpenAI

# ---------------------------- Helpers ----------------------------
class Cue:
    def __init__(self, start: int, end: int, text: str):
        self.start = start
        self.end = end
        self.text = text.strip()

    def to_dict(self):
        return {"start": self.start, "end": self.end, "text": self.text}

# Regex for junk cleanup
JUNK_RE = re.compile(r"(https?://\S+|www\.\S+|presented by|translator)", re.IGNORECASE)
SENT_END_RE = re.compile(r"[\.\!\?。！？…]+['\"]?$")

def load_srt(path: str) -> List[Cue]:
    subs = pysubs2.load(path)
    return [Cue(c.start, c.end, c.text) for c in subs]


def write_srt(cues: List[Cue], path: str) -> None:
    subs = pysubs2.SSAFile()
    for idx, c in enumerate(cues, start=1):
        subs.append(pysubs2.SSAEvent(start=c.start, end=c.end, text=c.text))
    subs.save(path)


def strip_junk(cues: List[Cue], window_ms: int = 30000) -> List[Cue]:
    if not cues:
        return []
    total_end = cues[-1].end
    cleaned = []
    for c in cues:
        if c.start < window_ms or c.end > total_end - window_ms:
            txt = JUNK_RE.sub("", c.text).strip()
            if txt:
                cleaned.append(Cue(c.start, c.end, txt))
        else:
            cleaned.append(c)
    return cleaned


def merge_utterances(cues: List[Cue], gap_ms: int = 500) -> List[Cue]:
    merged = []
    buf = []
    start = 0
    end = 0
    for c in cues:
        if not buf:
            start, end = c.start, c.end
        # if large gap, flush buffer
        if buf and c.start - end > gap_ms:
            merged.append(Cue(start, end, " ".join(buf)))
            buf = []
            start = c.start
        buf.append(c.text)
        end = c.end
        # if sentence ends, flush
        if SENT_END_RE.search(c.text):
            merged.append(Cue(start, end, " ".join(buf)))
            buf = []
    if buf:
        merged.append(Cue(start, end, " ".join(buf)))
    return merged


def extend_timestamps(cues: List[Cue]) -> List[Cue]:
    # Already cues cover utterance spans; nothing additional for now
    return cues

# ---------------------- LLM Alignment ----------------------
def get_client() -> OpenAI:
    cfg = json.loads(Path("settings.json").read_text()) if Path("settings.json").exists() else {}
    api_key = cfg.get("llm", {}).get("api_key")
    base = cfg.get("llm", {}).get("api_base", None)
    return OpenAI(api_key=api_key, base_url=base) if api_key else None


def call_llm_select_target(sl_text: str, tl_block: str, model: str = "gpt-4o") -> List[dict]:
    client = get_client()
    system = {"role": "system", "content": (
        "You are given a source utterance and a block of target language subtitle lines. "
        "Select the line or concatenation of lines from the target that best corresponds semantically to the source. "
        "Return a JSON array of objects with fields start, end, text matching the original target timestamps."
    )}
    user = {"role": "user", "content": json.dumps({"source": sl_text, "target_block": tl_block}, ensure_ascii=False)}
    resp = client.chat.completions.create(
        model=model,
        messages=[system, user],
        max_tokens=512,
        temperature=0.0,
    )
    out = resp.choices[0].message.content.strip()
    # sanitize fences
    if out.startswith("```"):
        out = "\n".join(out.splitlines()[1:-1])
    return json.loads(out)


def align_cues(sl_cues: List[Cue], tl_cues: List[Cue], context: int = 2) -> Tuple[List[Cue], List[Cue]]:
    new_sl = []
    new_tl = []
    for sl in sl_cues:
        # collect overlapping tl cues plus neighbors
        idxs = [i for i, t in enumerate(tl_cues) if not (t.end < sl.start - 1 or t.start > sl.end + 1)]
        # add context
        idxs = list(range(max(0, min(idxs or [0]) - context), min(len(tl_cues), max(idxs or [0]) + context + 1)))
        block = [t.to_dict() for i, t in enumerate(tl_cues) if i in idxs]
        tl_block = json.dumps(block, ensure_ascii=False)
        selected = call_llm_select_target(sl.text, tl_block)
        # apply source timestamps
        new_sl.append(Cue(sl.start, sl.end, sl.text))
        merged_text = " ".join(item['text'] for item in selected)
        new_tl.append(Cue(sl.start, sl.end, merged_text))
    return new_sl, new_tl

# ------------------------- Main -------------------------
def main(tl_path: str, sl_path: str, out_base: str = "final_synced"):
    # load
    tl = load_srt(tl_path)
    sl = load_srt(sl_path)
    # clean
    tl = strip_junk(tl)
    sl = strip_junk(sl)
    # utterances
    sl_utts = merge_utterances(sl)
    # align
    out_sl, out_tl = align_cues(sl_utts, tl)
    # write
    write_srt(out_tl, f"{out_base}_tl.srt")
    write_srt(out_sl, f"{out_base}_sl.srt")
    print("Alignment complete.")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('tl_file')
    p.add_argument('sl_file')
    p.add_argument('--out', default='final_synced')
    args = p.parse_args()
    main(args.tl_file, args.sl_file, out_base=args.out)
