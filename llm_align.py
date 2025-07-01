#!/usr/bin/env python3
import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from difflib import SequenceMatcher
from pypinyin import lazy_pinyin
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
    """Return ASCII/pinyin representation for rough cross-language matching."""
    # If the text contains Chinese characters convert them to pinyin for a crude
    # similarity check.  Other scripts simply pass through.
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        text = ' '.join(lazy_pinyin(text))
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


# ──────────────────────────────────────────────────────────────────────────────
# 5) Main routine
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sync_subtitles import make_cued, pair_subtitles, write_synced_subs
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

    tl_cues = make_cued(args.tl_file)
    sl_cues = make_cued(args.sl_file)

    raw_collapsed = pair_subtitles(tl_cues, sl_cues)
    llm_aligned = align_with_llm(raw_collapsed, batch_size=30)
    aligned = adjust_aligned_timings(llm_aligned)

    write_synced_subs(aligned, args.out, args.tl_code, args.sl_code)

    print(f"Done!  {args.out}_{args.tl_code}.srt + {args.out}_{args.sl_code}.srt generated.")
