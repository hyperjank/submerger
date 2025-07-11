#!/usr/bin/env python3
import os
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
from difflib import SequenceMatcher
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Load your API key from .env
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # look for a .env file in cwd or above

# Environment variables allow overriding the default LLM credentials and
# endpoint. By default we assume a local API (``LLM_API_BASE``) with
# ``LLM_API_KEY``. The historical ``DEEPSEEK_API_KEY`` points to the public
# DeepSeek service and can be enabled via ``--deepseek`` on the CLI.
api_key = os.getenv("LLM_API_KEY")
endpoint = os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
model = os.getenv("LLM_MODEL", "deepseek-chat")

# Instantiate the client only when a key is provided so library imports do not
# fail in environments without credentials (e.g. during testing).
client = OpenAI(api_key=api_key, base_url=endpoint) if api_key else None


def sanitize_json(text: str) -> str:
    """
    Remove Markdown fences (``` or ```json) from around a JSON payload.
    """
    # If it starts with ``` then strip the first and last lines
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        # drop the first line (``` or ```json)
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        # drop the last fence
        lines = lines[:-1]
    return "\n".join(lines).strip()


JUNK_RE = re.compile(
    r"(https?://\S+|www\.\S+|presented by|translator)", re.IGNORECASE
)


def regex_cleanup(cues: List[Dict], window_ms: int = 30000) -> List[Dict]:
    """Strip URLs and similar junk from the first/last ``window_ms`` of the file."""
    if not cues:
        return []

    total_end = cues[-1]["end_time"]
    cleaned: List[Dict] = []

    for cue in cues:
        if cue["start_time"] < window_ms or cue["end_time"] > total_end - window_ms:
            text = JUNK_RE.sub("", cue["text"]).strip()
            if not text:
                continue
            cleaned.append({
                "start_time": cue["start_time"],
                "end_time": cue["end_time"],
                "text": text,
            })
        else:
            cleaned.append(cue.copy())

    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# 1) Windowing to keep prompts small enough
# ──────────────────────────────────────────────────────────────────────────────

def windowed(chunks: List, size: int):
    """Yield (offset, sublist) for batching."""
    for i in range(0, len(chunks), size):
        yield i, chunks[i:i + size]


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


def needs_llm(tl_seg: Dict, sl_seg: Dict, *, time_tol: int = 700, sim_thr: float = 0.3) -> bool:
    """Return True if heuristic match fails and LLM should be consulted."""
    # If timestamps are roughly aligned we accept the pair immediately
    if abs(tl_seg['start_time'] - sl_seg['start_time']) <= time_tol and \
       abs(tl_seg['end_time'] - sl_seg['end_time']) <= time_tol:
        return False

    # Fallback: compare romanized text similarity
    return local_similarity(tl_seg['text'], sl_seg['text']) < sim_thr


# ──────────────────────────────────────────────────────────────────────────────
# 2) Call your local LLM endpoint for alignment
# ──────────────────────────────────────────────────────────────────────────────

def call_llm_for_alignment(tl_text: str, sl_text: str, model: str, max_tokens: int = 5) -> bool:
    """Ask the LLM if the source line is a translation of the target line."""

    if client is None:
        raise RuntimeError(
            "LLM client not configured; set LLM_API_KEY or DEEPSEEK_API_KEY"
        )

    system = {
        "role": "system",
        "content": (
            "You decide whether two lines are translations of each other. "
            "Respond only with 'Yes' or 'No'."
        ),
    }
    user = {
        "role": "user",
        "content": f"TL: {tl_text}\nSL: {sl_text}"
    }

    while True:
        resp = client.chat.completions.create(
            model=model,
            messages=[system, user],
            stream=False,
            max_tokens=max_tokens,
        )
        if resp and resp.choices and resp.choices[0].message:
            answer = resp.choices[0].message.content.strip().lower()
            if answer.startswith("y"):
                return True
            if answer.startswith("n"):
                return False


def sample_segments(cues: List[Dict], window_ms: int = 10000) -> List[Dict]:
    """Return cues from the first and last ``window_ms`` of the file."""

    if not cues:
        return []

    total_end = cues[-1]['end_time']
    selected = [
        {'start_time': c['start_time'], 'end_time': c['end_time'], 'text': c['text']}
        for c in cues
        if c['start_time'] < window_ms or c['end_time'] > total_end - window_ms
    ]
    return selected


def call_llm_for_cleanup(tl_sample: List[Dict], sl_sample: List[Dict], model: str, max_tokens: int = 200) -> str:
    """Ask the LLM to remove ads and similar junk from the provided samples."""

    if client is None:
        raise RuntimeError(
            "LLM client not configured; set LLM_API_KEY or DEEPSEEK_API_KEY"
        )

    system = {
        "role": "system",
        "content": (
            "You clean subtitle files. Strip advertisements, translator signatures, URLs "
            "and other extraneous text. Delete segments that are not part of the dialog or narration."
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps({"tl_sample": tl_sample, "sl_sample": sl_sample}, ensure_ascii=False),
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


def apply_cleanup(cues: List[Dict], original: List[Dict], cleaned: List[Dict]) -> List[Dict]:
    """Return ``cues`` with ``original`` segments replaced by ``cleaned``."""

    orig_keys = {(c['start_time'], c['end_time']) for c in original}
    clean_map = {(c['start_time'], c['end_time']): c for c in cleaned}
    updated: List[Dict] = []

    for cue in cues:
        key = (cue['start_time'], cue['end_time'])
        if key in orig_keys:
            cleaned_cue = clean_map.get(key)
            if cleaned_cue:
                text = cleaned_cue.get('text', '').strip()
                if text:
                    updated.append({
                        'start_time': cue['start_time'],
                        'end_time': cue['end_time'],
                        'text': text,
                    })
            # omitted cue means it was removed
        else:
            updated.append(cue)

    return updated


# ──────────────────────────────────────────────────────────────────────────────
# 4) Glue everything together
# ──────────────────────────────────────────────────────────────────────────────

def align_with_llm(
    collapsed: List[Dict],
    batch_size: int = 60,
    model: str = model,
    context: int = 2,
    sim_threshold: float = 0.3,
    time_tolerance: int = 700,
) -> List[Dict]:
    """Align subtitles using heuristics with optional LLM confirmation."""

    final: List[Dict] = []
    for seg in collapsed:
        tl_seg = {
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
            'text': seg['tl_text'],
        }
        sl_seg = {
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
            'text': seg['sl_text'],
        }

        if seg['tl_text'] and seg['sl_text'] and needs_llm(tl_seg, sl_seg, time_tol=time_tolerance, sim_thr=sim_threshold):
            try:
                if not call_llm_for_alignment(seg['tl_text'], seg['sl_text'], model=model):
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
            except Exception:
                # if the LLM call fails, fall back to heuristic result
                pass

        final.append(seg)

    # collapse any repeated adjacent segments from both paths
    collapsed_final = []
    for seg in final:
        last = collapsed_final[-1] if collapsed_final else None
        if last and last['tl_text'] == seg['tl_text'] and last['sl_text'] == seg['sl_text'] and last['end_time'] == seg['start_time']:
            last['end_time'] = seg['end_time']
        else:
            collapsed_final.append(seg)
    return collapsed_final


def adjust_aligned_timings(collapsed: List[Dict], time_tolerance: int = 700) -> List[Dict]:
    """Expand matching TL/SL pairs so their timestamps line up.

    If two segments contain translations of each other and their start/end
    times are within ``time_tolerance``, the shorter segment is lengthened to
    match the longer one.  Adjacent one-sided segments with the same text are
    absorbed into the expanded window.
    """

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

    return final


def semantic_align_cues(
    tl_cues: List[Dict],
    sl_cues: List[Dict],
    *,
    model: str = model,
    window_ms: int = 2000,
    sim_threshold: float = 0.6,
) -> List[Dict]:
    """Return segments where matching TL/SL lines share the TL timestamps."""

    aligned: List[Dict] = []
    used_sl = set()
    sl_index = 0

    for tl in tl_cues:
        match_idx = None
        j = sl_index
        while j < len(sl_cues) and sl_cues[j]["start_time"] <= tl["start_time"] + window_ms:
            sl = sl_cues[j]
            if j in used_sl:
                j += 1
                continue
            if abs(sl["start_time"] - tl["start_time"]) <= window_ms:
                if local_similarity(tl["text"], sl["text"]) >= sim_threshold:
                    match_idx = j
                    break
                if client and needs_llm(
                    {"start_time": tl["start_time"], "end_time": tl["end_time"], "text": tl["text"]},
                    {"start_time": sl["start_time"], "end_time": sl["end_time"], "text": sl["text"]},
                    time_tol=window_ms,
                    sim_thr=sim_threshold,
                ):
                    try:
                        if call_llm_for_alignment(tl["text"], sl["text"], model=model):
                            match_idx = j
                            break
                    except Exception:
                        pass
            if sl["start_time"] > tl["start_time"] + window_ms:
                break
            j += 1

        if match_idx is not None:
            sl = sl_cues[match_idx]
            used_sl.add(match_idx)
            sl_index = match_idx + 1
            aligned.append({
                "start_time": tl["start_time"],
                "end_time": tl["end_time"],
                "tl_text": tl["text"],
                "sl_text": sl["text"],
            })
        else:
            aligned.append({
                "start_time": tl["start_time"],
                "end_time": tl["end_time"],
                "tl_text": tl["text"],
                "sl_text": "",
            })

    for idx, sl in enumerate(sl_cues):
        if idx not in used_sl:
            aligned.append({
                "start_time": sl["start_time"],
                "end_time": sl["end_time"],
                "tl_text": "",
                "sl_text": sl["text"],
            })

    aligned.sort(key=lambda s: s["start_time"])
    return adjust_aligned_timings(aligned, time_tolerance=window_ms)


# ──────────────────────────────────────────────────────────────────────────────
# 5) Main routine
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sync_subtitles import (
        make_cued,
        dedupe_cues,
        pair_subtitles,
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

    if args.deepseek:
        # override globals to use the DeepSeek service
        api_key = os.getenv("DEEPSEEK_API_KEY")
        endpoint = "https://api.deepseek.com/v1"
        model = os.getenv("LLM_MODEL", "deepseek-chat")
        client = OpenAI(api_key=api_key, base_url=endpoint) if api_key else None

    tl_cues = dedupe_cues(regex_cleanup(make_cued(args.tl_file)))
    sl_cues = dedupe_cues(regex_cleanup(make_cued(args.sl_file)))

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

    # First merge both tracks into a unified timeline so gaps on either side are
    # preserved for the alignment step.
    collapsed = pair_subtitles(tl_cues, sl_cues)

    # Run the semantic/heuristic alignment over the paired timeline.
    aligned = align_with_llm(collapsed, model=model)
    aligned = adjust_aligned_timings(aligned)

    write_synced_subs(aligned, args.out, args.tl_code, args.sl_code)

    print(f"Done!  {args.out}_{args.tl_code}.srt + {args.out}_{args.sl_code}.srt generated.")
