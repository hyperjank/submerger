#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path
from typing import List, Dict
from sync_subtitles import SubtitleCue
from difflib import SequenceMatcher
from openai import OpenAI

# Global model and client references populated by `get_client`.
model = "deepseek-chat"
client = None
_config = None


def load_config() -> dict:
    """Load settings.json from this directory."""
    global _config
    if _config is None:
        path = Path(__file__).with_name("settings.json")
        if path.exists():
            with path.open() as f:
                _config = json.load(f)
        else:
            _config = {}
    return _config


def get_client() -> OpenAI | None:
    """Load LLM config and return a cached OpenAI client."""
    global client, model
    if client is None:
        cfg = load_config().get("llm", {})
        api_key = cfg.get("api_key")
        endpoint = cfg.get("api_base", "http://localhost:1234/v1")
        model = cfg.get("model", "deepseek-chat")
        client = OpenAI(api_key=api_key, base_url=endpoint) if api_key else None
    return client


def sanitize_json(text: str) -> str:
    """
    Remove Markdown fences (``` or ```json) from around a JSON payload.
    """
    # If it starts with ``` then strip the first and last lines
    lines = text.strip().splitlines()
    # Remove any lines equal to "<think>" or "</think>"
    lines = [line for line in lines if line.strip() not in ("<think>", "</think>")]
    # Remove leading/trailing blank lines introduced by the above filter
    lines = "\n".join(lines).strip().splitlines()
    # If it starts with ``` after removing think tags, drop the first line
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # And drop the last fence if present
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove any <think>...</think> blocks and stray tags."""
    text = THINK_BLOCK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


JUNK_RE = re.compile(
    r"(https?://\S+|www\.\S+|presented by|translator)", re.IGNORECASE
)


def regex_cleanup(cues: List[SubtitleCue], window_ms: int = 30000) -> List[SubtitleCue]:
    """Strip URLs and similar junk from the first/last ``window_ms`` of the file."""
    if not cues:
        return []

    total_end = cues[-1].end_time
    cleaned: List[SubtitleCue] = []
    removed = 0

    for cue in cues:
        if cue.start_time < window_ms or cue.end_time > total_end - window_ms:
            text = JUNK_RE.sub("", cue.text).strip()
            if not text:
                removed += 1
                continue
            cleaned.append(SubtitleCue(cue.start_time, cue.end_time, text))
        else:
            cleaned.append(SubtitleCue(cue.start_time, cue.end_time, cue.text))

    if removed:
        print(f"regex_cleanup: removed {removed} junk cues")
    return cleaned


# ------------------------------------------------------------------------------
# Heuristic helpers
# ------------------------------------------------------------------------------

def _normalized(text: str) -> str:
    """Return a simplified ASCII representation for rough matching."""
    text = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text)
    return ' '.join(text.split())


def local_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio after basic normalization."""
    return SequenceMatcher(None, _normalized(a), _normalized(b)).ratio()


def needs_llm(tl_seg: SubtitleCue, sl_seg: SubtitleCue, *, time_tol: int = 700, sim_thr: float = 0.3) -> bool:
    """Return True if heuristic match fails and LLM should be consulted."""
    # If timestamps are roughly aligned we accept the pair immediately
    if abs(tl_seg.start_time - sl_seg.start_time) <= time_tol and \
       abs(tl_seg.end_time - sl_seg.end_time) <= time_tol:
        return False

    # Fallback: compare romanized text similarity
    return local_similarity(tl_seg.text, sl_seg.text) < sim_thr

# ------------------------------------------------------------------------------
# Sentence helpers
# ------------------------------------------------------------------------------

SENT_END_RE = re.compile(r"[.!?。！？…]+['\"]?$")


def combine_into_sentences(cues: List[SubtitleCue], gap_ms: int = 500) -> List[SubtitleCue]:
    """Merge cues until they form complete sentences/utterances."""

    if not cues:
        return []

    merged: List[SubtitleCue] = []
    buf: List[str] = []
    start = cues[0].start_time
    end = cues[0].end_time

    for cue in cues:
        if not buf:
            start = cue.start_time
        elif cue.start_time - end > gap_ms:
            merged.append(SubtitleCue(start, end, " ".join(buf)))
            buf = []
            start = cue.start_time

        buf.append(cue.text)
        end = cue.end_time

        if SENT_END_RE.search(cue.text.strip()):
            merged.append(SubtitleCue(start, end, " ".join(buf)))
            buf = []

    if buf:
        merged.append(SubtitleCue(start, end, " ".join(buf)))

    print(f"combine_into_sentences: {len(cues)} cues -> {len(merged)} sentences")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 2) Call your local LLM endpoint for alignment
# ──────────────────────────────────────────────────────────────────────────────

def call_llm_for_alignment(tl_text: str, sl_text: str, model: str, max_tokens: int = 5) -> bool:
    """Ask the LLM if the source line is a translation of the target line."""

    client = get_client()
    if client is None:
        raise RuntimeError(
            "LLM client not configured; set LLM_API_KEY or DEEPSEEK_API_KEY"
        )

    system = {
        "role": "system",
        "content": (
            "You decide whether two lines are translations of each other. "
            "Respond only with Yes or No."
        ),
    }
    user = {
        "role": "user",
        "content": f"TL: {tl_text}\nSL: {sl_text}"
    }

    unknown = 0
    while True:
        resp = client.chat.completions.create(
            model=model,
            messages=[system, user],
            stream=False,
            max_tokens=max_tokens,
        )
        answer = ""
        if resp and resp.choices and resp.choices[0].message:
            answer = resp.choices[0].message.content.strip().lower()
            if answer.startswith("y"):
                return True
            if answer.startswith("n"):
                return False
        unknown += 1
        if unknown >= 3:
            print(f"Unrecognized LLM response: {answer!r} - giving up")
            break
    return False


def call_llm_for_translation(
    sl_text: str,
    sl_code: str,
    tl_code: str,
    model: str,
    *,
    tl_block: str | None = None,
    max_tokens: int = 500,
) -> str:
    """Ask the LLM to translate ``sl_text`` or select the matching TL line."""

    client = get_client()
    if client is None:
        raise RuntimeError(
            "LLM client not configured; set LLM_API_KEY or DEEPSEEK_API_KEY"
        )

    if tl_block:
        instruction = (
            f"You translate from {sl_code} to {tl_code}. If possible, "
            "pick and return only the line from the provided target text that matches the source. "
            "Otherwise give your own translation."
        )
        user_content = f"Source: {sl_text}\n\nTarget options:\n{tl_block}"
    else:
        instruction = (
            f"You translate from {sl_code} to {tl_code}. Respond only with the translation."
        )
        user_content = sl_text

    system = {
        "role": "system",
        "content": instruction,
    }
    user = {"role": "user", "content": user_content}

    resp = client.chat.completions.create(
        model=model,
        messages=[system, user],
        stream=False,
        max_tokens=max_tokens,
    )

    if resp and resp.choices and resp.choices[0].message:
        return strip_think(resp.choices[0].message.content.strip())

    return ""


def sample_segments(cues: List[SubtitleCue], window_ms: int = 10000) -> List[SubtitleCue]:
    """Return cues from the first and last ``window_ms`` of the file."""

    if not cues:
        return []

    total_end = cues[-1].end_time
    selected = [
        SubtitleCue(c.start_time, c.end_time, c.text)
        for c in cues
        if c.start_time < window_ms or c.end_time > total_end - window_ms
    ]
    return selected


def call_llm_for_cleanup(tl_sample: List[SubtitleCue], sl_sample: List[SubtitleCue], model: str, max_tokens: int = 1000) -> str:
    """Ask the LLM to remove ads and similar junk from the provided samples."""

    client = get_client()
    if client is None:
        raise RuntimeError(
            "LLM client not configured; set LLM_API_KEY or DEEPSEEK_API_KEY"
        )

    system = {
        "role": "system",
        "content": (
            "You clean subtitle files. Strip advertisements, translator signatures, URLs "
            "and other extraneous text. Delete segments that are not part of the dialog or narration. /nothink"
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps({
            "tl_sample": [c.to_dict() for c in tl_sample],
            "sl_sample": [c.to_dict() for c in sl_sample],
        }, ensure_ascii=False),
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[system, user],
        stream=False,
        max_tokens=max_tokens,
    )

    if resp and resp.choices and resp.choices[0].message:
        return resp.choices[0].message.content

    return ""


def apply_cleanup(cues: List[SubtitleCue], original: List[SubtitleCue], cleaned: List[Dict]) -> List[SubtitleCue]:
    """Return ``cues`` with ``original`` segments replaced by ``cleaned``."""

    orig_keys = {(c.start_time, c.end_time) for c in original}
    clean_map = {(c['start_time'], c['end_time']): c for c in cleaned}
    updated: List[SubtitleCue] = []

    for cue in cues:
        key = (cue.start_time, cue.end_time)
        if key in orig_keys:
            cleaned_cue = clean_map.get(key)
            if cleaned_cue:
                text = cleaned_cue.get('text', '').strip()
                if text:
                    updated.append(SubtitleCue(cue.start_time, cue.end_time, text))
            # omitted cue means it was removed
        else:
            updated.append(cue)

    return updated


# ──────────────────────────────────────────────────────────────────────────────
# 4) Glue everything together
# ──────────────────────────────────────────────────────────────────────────────

def align_with_llm(
    collapsed: List[Dict],
    model: str = model,
    context: int = 3,
    sim_threshold: float = 0.3,
    time_tolerance: int = 700,
) -> List[Dict]:
    """Align subtitles using heuristics with optional LLM confirmation."""

    print("Starting alignment phase...")
    final: List[Dict] = []
    for seg in collapsed:
        print(
            f"\nSegment {seg['start_time']}-{seg['end_time']}: TL='{seg['tl_text']}' | SL='{seg['sl_text']}'"
        )
        tl_seg = SubtitleCue(seg['start_time'], seg['end_time'], seg['tl_text'])
        sl_seg = SubtitleCue(seg['start_time'], seg['end_time'], seg['sl_text'])

        if seg['tl_text'] and seg['sl_text'] and needs_llm(tl_seg, sl_seg, time_tol=time_tolerance, sim_thr=sim_threshold):
            print("  Heuristic mismatch -> consulting LLM")
            try:
                if not call_llm_for_alignment(seg['tl_text'], seg['sl_text'], model=model):
                    print("  LLM rejected pair - splitting segment")
                    # treat as two separate segments if LLM says "No"
                    final.append({
                        'start_time': seg['start_time'],
                        'end_time': seg['end_time'],
                        'tl_text': seg['tl_text'],
                        'sl_text': '',
                    })
                    final.append({
                        'start_time': seg['start_time'],
                        'end_time': seg['end_time'],
                        'tl_text': '',
                        'sl_text': seg['sl_text'],
                    })
                    continue
            except Exception as exc:
                # if the LLM call fails, fall back to heuristic result
                print(f"  LLM call failed: {exc}, keeping heuristic result")

        final.append(seg)

    # collapse any repeated adjacent segments from both paths
    collapsed_final = []
    for seg in final:
        last = collapsed_final[-1] if collapsed_final else None
        if last and last['tl_text'] == seg['tl_text'] and last['sl_text'] == seg['sl_text'] and last['end_time'] == seg['start_time']:
            last['end_time'] = seg['end_time']
        else:
            collapsed_final.append(seg)
    print(f"Alignment phase produced {len(collapsed_final)} segments")
    return collapsed_final


def adjust_aligned_timings(collapsed: List[Dict], time_tolerance: int = 700) -> List[Dict]:
    """Expand matching TL/SL pairs so their timestamps line up.

    If two segments contain translations of each other and their start/end
    times are within ``time_tolerance``, the shorter segment is lengthened to
    match the longer one.  Adjacent one-sided segments with the same text are
    absorbed into the expanded window.
    """

    print("\nAdjusting timings to merge neighbouring segments...")
    adjusted: List[Dict] = []
    i = 0
    while i < len(collapsed):
        seg = collapsed[i]
        if seg['tl_text'] and seg['sl_text']:
            start = seg['start_time']
            end = seg['end_time']

            # absorb previous segments with same text on either side
            while adjusted:
                prev = adjusted[-1]
                if prev['sl_text'] == '' and prev['tl_text'] == seg['tl_text'] and start - prev['start_time'] <= time_tolerance:
                    start = prev['start_time']
                    adjusted.pop()
                elif prev['tl_text'] == '' and prev['sl_text'] == seg['sl_text'] and start - prev['start_time'] <= time_tolerance:
                    start = prev['start_time']
                    adjusted.pop()
                else:
                    break

            j = i + 1
            while j < len(collapsed):
                nxt = collapsed[j]
                if nxt['tl_text'] == seg['tl_text'] and nxt['sl_text'] == '' and nxt['end_time'] - end <= time_tolerance:
                    end = nxt['end_time']
                    j += 1
                elif nxt['tl_text'] == '' and nxt['sl_text'] == seg['sl_text'] and nxt['end_time'] - end <= time_tolerance:
                    end = nxt['end_time']
                    j += 1
                else:
                    break

            adjusted.append({
                'start_time': start,
                'end_time': end,
                'tl_text': seg['tl_text'],
                'sl_text': seg['sl_text'],
            })
            print(f"  Expanded pair to {start}-{end}")
            i = j
        else:
            adjusted.append(seg)
            i += 1

    # collapse again after expansion
    final: List[Dict] = []
    for seg in adjusted:
        last = final[-1] if final else None
        if last and last['tl_text'] == seg['tl_text'] and last['sl_text'] == seg['sl_text'] and last['end_time'] == seg['start_time']:
            last['end_time'] = seg['end_time']
        else:
            final.append(seg)
    print(f"Timing adjustment produced {len(final)} segments")
    return final


def sentence_level_align(
    tl_cues: List[SubtitleCue],
    sl_cues: List[SubtitleCue],
    *,
    tl_code: str = "tl",
    sl_code: str = "sl",
    model: str = model,
    context: int = 3,
    sim_threshold: float = 0.6,
) -> List[Dict]:
    """Align subtitle cues at the sentence level using fuzzy matching."""

    client = get_client()

    tl_sent = combine_into_sentences(tl_cues)
    sl_sent = combine_into_sentences(sl_cues)
    print(
        f"\nSentence alignment on {len(tl_sent)} TL sentences and {len(sl_sent)} SL sentences"
    )

    aligned: List[Dict] = []
    used = [False] * len(tl_sent)
    j = 0

    for idx, sl in enumerate(sl_sent):
        print(f"\nSL[{idx}] '{sl.text}'")
        match_start = None
        match_end = None
        cand_text = ""

        search_start = max(0, j - context)
        search_end = min(len(tl_sent), j + context + 1)

        for start in range(search_start, search_end):
            accum = ""
            for end in range(start, min(len(tl_sent), start + context + 1)):
                accum = (accum + " " + tl_sent[end].text).strip()
                if local_similarity(accum, sl.text) >= sim_threshold:
                    match_start = start
                    match_end = end
                    cand_text = accum
                    break
                if client and needs_llm(
                    SubtitleCue(0, 0, accum),
                    SubtitleCue(0, 0, sl.text),
                    time_tol=999999,
                    sim_thr=sim_threshold,
                ):
                    try:
                        if call_llm_for_alignment(accum, sl["text"], model=model):
                            match_start = start
                            match_end = end
                            cand_text = accum
                            break
                    except Exception as exc:
                        print(f"  LLM alignment failed: {exc}")
            if match_start is not None:
                break

        if match_start is not None and match_end is not None:
            print(f"  Matched TL[{match_start}:{match_end}] -> '{cand_text}'")
            aligned.append(
                {
                    "start_time": sl.start_time,
                    "end_time": sl.end_time,
                    "tl_text": cand_text,
                    "sl_text": sl.text,
                }
            )
            for k in range(match_start, match_end + 1):
                used[k] = True
            j = match_end + 1
        else:
            print("  No TL match found -> translating")
            translation = ""
            if client:
                try:
                    window = 3000
                    context_lines = [
                        t.text
                        for t in tl_sent
                        if t.start_time < sl.end_time + window
                        and t.end_time > sl.start_time - window
                    ]
                    tl_block = "\n".join(context_lines)
                    translation = call_llm_for_translation(
                        sl.text,
                        sl_code,
                        tl_code,
                        model=model,
                        tl_block=tl_block,
                    )
                except Exception as exc:
                    print(f"  Translation failed: {exc}")
            aligned.append(
                {
                    "start_time": sl.start_time,
                    "end_time": sl.end_time,
                    "tl_text": translation,
                    "sl_text": sl.text,
                }
            )

    for idx, tl in enumerate(tl_sent):
        if not used[idx]:
            aligned.append(
                {
                    "start_time": tl.start_time,
                    "end_time": tl.end_time,
                    "tl_text": tl.text,
                    "sl_text": "",
                }
            )

    aligned.sort(key=lambda s: s["start_time"])
    return aligned


def semantic_align_cues(
    tl_cues: List[SubtitleCue],
    sl_cues: List[SubtitleCue],
    *,
    model: str = model,
    window_ms: int = 2000,
    sim_threshold: float = 0.6,
) -> List[Dict]:
    """Return sentence-aligned segments with matching TL timestamps."""

    aligned = sentence_level_align(
        tl_cues,
        sl_cues,
        tl_code="tl",
        sl_code="sl",
        model=model,
        context=3,
        sim_threshold=sim_threshold,
    )

    return adjust_aligned_timings(aligned, time_tolerance=window_ms)


# ──────────────────────────────────────────────────────────────────────────────
# 5) Main routine
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sync_subtitles import (
        make_cued,
        dedupe_cues,
        write_synced_subs,
    )
    import argparse

    parser = argparse.ArgumentParser(description="Align two subtitle files via an LLM")
    parser.add_argument("tl_file", help="Target language subtitle file")
    parser.add_argument("sl_file", help="Source language subtitle file")
    parser.add_argument("--tl-code", default="tl", help="language code for the target language")
    parser.add_argument("--sl-code", default="sl", help="language code for the source language")
    parser.add_argument("--out", default="final_synced", help="base path for output files")
    parser.add_argument(
        "--deepseek",
        action="store_true",
        help="Use the DeepSeek API instead of the local LLM endpoint",
    )

    args = parser.parse_args()

    client = get_client()

    if args.deepseek:
        # override globals to use the DeepSeek service
        api_key = os.getenv("DEEPSEEK_API_KEY")
        endpoint = "https://api.deepseek.com/v1"
        model = load_config().get("llm", {}).get("model", "deepseek-chat")
        client = OpenAI(api_key=api_key, base_url=endpoint) if api_key else None

    tl_cues = dedupe_cues(make_cued(args.tl_file))
    sl_cues = dedupe_cues(regex_cleanup(make_cued(args.sl_file)))

    print(f"Loaded {len(tl_cues)} TL cues and {len(sl_cues)} SL cues after cleanup")

    if client:
        try:
            tl_sample = sample_segments(tl_cues, window_ms=30000)
            sl_sample = sample_segments(sl_cues, window_ms=30000)
            response = call_llm_for_cleanup(tl_sample, sl_sample, model=model)
            print("LLM cleanup suggestion:\n" + response)
            try:
                cleaned = json.loads(sanitize_json(response))
                tl_cues = apply_cleanup(tl_cues, tl_sample, cleaned.get('tl_sample', []))
                sl_cues = apply_cleanup(sl_cues, sl_sample, cleaned.get('sl_sample', []))
            except Exception as exc:
                print(f"LLM cleanup parse failed: {exc}")
        except Exception as exc:
            print(f"LLM cleanup failed: {exc}")

    # Align at the sentence level using fuzzy matching and the LLM.
    aligned = sentence_level_align(
        tl_cues,
        sl_cues,
        tl_code=args.tl_code,
        sl_code=args.sl_code,
        model=model,
    )
    aligned = adjust_aligned_timings(aligned)

    write_synced_subs(aligned, args.out, args.tl_code, args.sl_code)

    print(f"Done!  {args.out}_{args.tl_code}.srt + {args.out}_{args.sl_code}.srt generated.")
