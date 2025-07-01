#!/usr/bin/env python3

import pysubs2
from typing import List, Dict


def make_cued(source_subtitle):
    subs = pysubs2.load(source_subtitle)
    cues = []
    for ev in subs.events:
        # skip cues with no visible text after stripping markup
        if not ev.plaintext.strip():
            continue
        cues.append({
            'start_time': ev.start,
            'end_time': ev.end,
            'text': ev.plaintext.strip(),
        })
    return sorted(cues, key=lambda c: c['start_time'])


def dedupe_cues(cues: List[Dict]) -> List[Dict]:
    """Return a new list with consecutive duplicate texts removed."""
    deduped: List[Dict] = []
    last_text = None
    for cue in cues:
        if cue['text'] == last_text:
            continue
        deduped.append(cue)
        last_text = cue['text']
    return deduped


def pair_subtitles(tl_cued, sl_cued):

    """Return a merged timeline of TL/SL cues.

    The input cue lists should already be sorted by ``start_time``.  The
    function walks both lists with independent indexes so every cue is visited
    only once.  This linear pass is significantly faster than repeatedly
    searching the lists for the active cue at each point in the timeline.
    """

    # Collect all boundaries that define the timeline
    times = {t for cue in tl_cued for t in (cue["start_time"], cue["end_time"])}
    times.update(t for cue in sl_cued for t in (cue["start_time"], cue["end_time"]))

    timeline = sorted(times)

    paired_subs = []
    i = j = 0
    tl_cue = tl_cued[i] if tl_cued else None
    sl_cue = sl_cued[j] if sl_cued else None
    tl_text = ""
    sl_text = ""

    for t0, t1 in zip(timeline, timeline[1:]):
        # Advance TL index until the cue covering ``t0`` is found
        while tl_cue and tl_cue["end_time"] <= t0:
            i += 1
            tl_cue = tl_cued[i] if i < len(tl_cued) else None
        if tl_cue and tl_cue["start_time"] <= t0 < tl_cue["end_time"]:
            tl_text = tl_cue["text"]
        else:
            tl_text = ""

        # Advance SL index until the cue covering ``t0`` is found
        while sl_cue and sl_cue["end_time"] <= t0:
            j += 1
            sl_cue = sl_cued[j] if j < len(sl_cued) else None
        if sl_cue and sl_cue["start_time"] <= t0 < sl_cue["end_time"]:
            sl_text = sl_cue["text"]
        else:
            sl_text = ""

        if not (tl_text or sl_text):
            continue

        last = paired_subs[-1] if paired_subs else None
        if last and last["tl_text"] == tl_text and last["sl_text"] == sl_text:
            last["end_time"] = t1
        else:
            paired_subs.append({
                "start_time": t0,
                "end_time": t1,
                "tl_text": tl_text,
                "sl_text": sl_text,
            })

    return paired_subs


def write_synced_subs(collapsed, outpath, tl_lang_code, sl_lang_code):
    tl_subs = pysubs2.SSAFile()
    sl_subs = pysubs2.SSAFile()

    for seg in collapsed:
        tl_subs.events.append(
            pysubs2.SSAEvent(
                start=seg["start_time"],
                end=seg["end_time"],
                text=seg["tl_text"]
            )
        )
        sl_subs.events.append(
            pysubs2.SSAEvent(
                start=seg["start_time"],
                end=seg["end_time"],
                text=seg["sl_text"]
            )
        )

    tl_subs.save(f"{outpath}_{tl_lang_code}.srt")
    sl_subs.save(f"{outpath}_{sl_lang_code}.srt")


def write_scripts(collapsed: List[Dict], outpath: str, tl_lang_code: str, sl_lang_code: str) -> None:
    """Write plain text scripts for both languages without duplicate lines."""
    tl_lines: List[str] = []
    sl_lines: List[str] = []
    last_tl = None
    last_sl = None
    for seg in collapsed:
        if seg['tl_text'] and seg['tl_text'] != last_tl:
            tl_lines.append(seg['tl_text'])
            last_tl = seg['tl_text']
        if seg['sl_text'] and seg['sl_text'] != last_sl:
            sl_lines.append(seg['sl_text'])
            last_sl = seg['sl_text']


    with open(f"{outpath}_{tl_lang_code}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(tl_lines))

    with open(f"{outpath}_{sl_lang_code}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sl_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pair two subtitle files")
    parser.add_argument("tl_file", help="Target language subtitle file")
    parser.add_argument("sl_file", help="Source language subtitle file")
    parser.add_argument("--tl-code", default="tl", help="language code for target language")
    parser.add_argument("--sl-code", default="sl", help="language code for source language")
    parser.add_argument("--out", default="synced_subs", help="base path for output files")

    args = parser.parse_args()

    tl_cues = dedupe_cues(make_cued(args.tl_file))
    sl_cues = dedupe_cues(make_cued(args.sl_file))
    paired_subs = pair_subtitles(tl_cues, sl_cues)
    write_synced_subs(paired_subs, args.out, args.tl_code, args.sl_code)
    write_scripts(paired_subs, args.out, args.tl_code, args.sl_code)


