#!/usr/bin/env python3
import os
import json
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from difflib import SequenceMatcher
from pypinyin import lazy_pinyin
import pysubs2
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
endpoint = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
model = os.getenv("LLM_MODEL", "deepseek-chat")

# Instantiate the client only when a key is provided so library imports do not
# fail in environments without credentials (e.g. during testing).
client = OpenAI(api_key=api_key, base_url=endpoint) if api_key else None
# ──────────────────────────────────────────────────────────────────────────────
# 1) Utilities to write out SRTs
# ──────────────────────────────────────────────────────────────────────────────

def write_synced_subs(segments: List[Dict], base_out: str, tl_lang: str, sl_lang: str):
    tl_subs, sl_subs = pysubs2.SSAFile(), pysubs2.SSAFile()
    for seg in segments:
        tl_subs.events.append(pysubs2.SSAEvent(
            start=seg['start_time'], end=seg['end_time'], text=seg['tl_text']
        ))
        sl_subs.events.append(pysubs2.SSAEvent(
            start=seg['start_time'], end=seg['end_time'], text=seg['sl_text']
        ))
    tl_subs.save(f"{base_out}_{tl_lang}.srt")
    sl_subs.save(f"{base_out}_{sl_lang}.srt")

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
# 2) Windowing to keep prompts small enough
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
# 3) Call your local LLM endpoint for alignment
# ──────────────────────────────────────────────────────────────────────────────

def call_llm_for_alignment(
    tl_texts: List[str],
    sl_texts: List[str],
    model: str,
    max_tokens: int = 8192,
) -> List[Dict]:
    """Ask the LLM to align and re-time each pair of subtitle chunks.

    Returns a list of dictionaries each containing:
      tl_idx (int), sl_idx (int),
      start_time (int ms), end_time (int ms).
    """
    system = {
        "role": "system",
        "content": (
            "You are a subtitle alignment assistant. Given two lists of subtitle "
            "chunks in different languages, return a JSON array describing the "
            "aligned pairs with these fields:\n"
            "- tl_idx and sl_idx (original indices)\n"
            "- start_time and end_time in milliseconds (new display window)\n"
            "Ensure both texts appear together and strip any ads or URLs."
        )
    }
    user = {
        "role": "user",
        "content": (
            "Target language chunks (index: text):\n"
            + "\n".join(f"{i}: {t}" for i, t in enumerate(tl_texts))
            + "\n\nSource language chunks (index: text):\n"
            + "\n".join(f"{j}: {t}" for j, t in enumerate(sl_texts))
            + "\n\nRespond with ONLY a JSON array like:\n"
            "[{\"tl_idx\":0,\"sl_idx\":2,\"start_time\":1234,\"end_time\":5678}, …]"
        )
    }

    if client is None:
        raise RuntimeError(
            "LLM client not configured; set LLM_API_KEY or DEEPSEEK_API_KEY"
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[system, user],
        stream=False,
        max_tokens=max_tokens,
    )
    content = sanitize_json(resp.choices[0].message.content)
    try:
        items = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from LLM after sanitization:\n{content}")

    # validate and return
    aligned = []
    for it in items:
        for k in ("tl_idx", "sl_idx", "start_time", "end_time"):
            if k not in it:
                raise KeyError(f"Alignment object missing `{k}`: {it}")
        aligned.append({
            "tl_idx":     int(it["tl_idx"]),
            "sl_idx":     int(it["sl_idx"]),
            "start_time": int(it["start_time"]),
            "end_time":   int(it["end_time"]),
        })
    return aligned


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
    """Align using heuristics first, falling back to the LLM when needed."""

    final = []
    idx = 0
    while idx < len(collapsed):
        seg = collapsed[idx]

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

        if seg['tl_text'] and seg['sl_text'] and not needs_llm(tl_seg, sl_seg, time_tol=time_tolerance, sim_thr=sim_threshold):
            # heuristic says these already align
            final.append(seg)
            idx += 1
            continue

        # collect a window of lines around the mismatch and ask the LLM
        start = max(0, idx - context)
        end = min(len(collapsed), idx + context + 1)

        tl_window = [s['tl_text'] for s in collapsed[start:end]]
        sl_window = [s['sl_text'] for s in collapsed[start:end]]
        results = call_llm_for_alignment(tl_window, sl_window, model=model)

        for r in results:
            a = collapsed[start + r['tl_idx']]
            b = collapsed[start + r['sl_idx']]
            final.append({
                'start_time': r['start_time'],
                'end_time': r['end_time'],
                'tl_text': a['tl_text'],
                'sl_text': b['sl_text'],
            })

        idx = end

    # collapse any repeated adjacent segments from both paths
    collapsed_final = []
    for seg in final:
        last = collapsed_final[-1] if collapsed_final else None
        if last and last['tl_text'] == seg['tl_text'] and last['sl_text'] == seg['sl_text'] and last['end_time'] == seg['start_time']:
            last['end_time'] = seg['end_time']
        else:
            collapsed_final.append(seg)
    return collapsed_final


# ──────────────────────────────────────────────────────────────────────────────
# 5) Main routine
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sync_subtitles import make_cued, pair_subtitles
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

    write_synced_subs(llm_aligned, args.out, args.tl_code, args.sl_code)

    print(f"Done!  {args.out}_{args.tl_code}.srt + {args.out}_{args.sl_code}.srt generated.")
