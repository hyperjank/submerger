#!/usr/bin/env python3
import os
import json
from typing import List, Tuple, Dict
from dotenv import load_dotenv
import pysubs2
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Load your API key from .env
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # look for a .env file in cwd or above
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not set in .env")
endpoint = "https://api.deepseek.com/v1"
model = "deepseek-chat"
client = OpenAI(api_key=api_key, base_url=endpoint)
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


# ──────────────────────────────────────────────────────────────────────────────
# 3) Call your local LLM endpoint for alignment
# ──────────────────────────────────────────────────────────────────────────────

def call_llm_for_alignment(
    tl_texts: List[str],
    sl_texts: List[str],
    model: str,
    max_tokens: int = 8192,
) -> List[Dict]:
    """
    Ask the LLM to align and re-time each Chinese–English chunk.
    Returns a list of dicts, each with keys:
      tl_idx (int), sl_idx (int),
      start_time (int ms), end_time (int ms).
    """
    system = {
        "role": "system",
        "content": (
            "You are a subtitle alignment assistant.  Given two lists of subtitle chunks, "
            "for each Chinese–English pair return a JSON array of objects with these fields:\n"
            "- tl_idx and sl_idx (original indices)\n"
            "- start_time and end_time in milliseconds (new display window)\n"
            "Ensure both texts appear together and strip any ads or URLs."
        )
    }
    user = {
        "role": "user",
        "content": (
            "Chinese chunks (index: text):\n"
            + "\n".join(f"{i}: {t}" for i, t in enumerate(tl_texts))
            + "\n\nEnglish chunks (index: text):\n"
            + "\n".join(f"{j}: {t}" for j, t in enumerate(sl_texts))
            + "\n\nRespond with ONLY a JSON array like:\n"
            "[{\"tl_idx\":0,\"sl_idx\":2,\"start_time\":1234,\"end_time\":5678}, …]"
        )
    }

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
    model: str = model
) -> List[Dict]:
    """
    Uses the LLM-returned start/end times instead of min/max, and rebuilds
    each segment with those times.
    """
    tl_texts = [seg['tl_text'] for seg in collapsed]
    sl_texts = [seg['sl_text'] for seg in collapsed]
    raw_aligned = []

    for offset, tl_window in windowed(tl_texts, batch_size):
        sl_window = sl_texts[offset:offset+batch_size]
        results = call_llm_for_alignment(tl_window, sl_window, model=model)

        for r in results:
            a = collapsed[offset + r["tl_idx"]]
            b = collapsed[offset + r["sl_idx"]]
            raw_aligned.append({
                "start_time": r["start_time"],
                "end_time":   r["end_time"],
                "tl_text":    a["tl_text"],
                "sl_text":    b["sl_text"],
            })

    # final collapse of any adjacent identical entries
    final = []
    for seg in raw_aligned:
        last = final[-1] if final else None
        if last and \
           last['tl_text'] == seg['tl_text'] and \
           last['sl_text'] == seg['sl_text'] and \
           last['end_time'] == seg['start_time']:
            last['end_time'] = seg['end_time']
        else:
            final.append(seg)
    return final


# ──────────────────────────────────────────────────────────────────────────────
# 5) Main routine
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sync_subtitles import make_cued, pair_subtitles  # adapt import

    # 1. load your two original files
    tl_cues = make_cued("synced_subs_zh.srt")
    sl_cues = make_cued("synced_subs_en.srt")

    # 2. get the rough collapsed timeline
    raw_collapsed = pair_subtitles(tl_cues, sl_cues)

    # 3. ask the LLM to refine that alignment
    llm_aligned = align_with_llm(raw_collapsed, batch_size=30)

    # 4. write out the final synced SRTs
    write_synced_subs(llm_aligned, "final_synced", "zh", "en")

    print("Done!  final_synced_zh.srt  +  final_synced_en.srt generated.")
