#!/usr/bin/env python3

import pysubs2


def make_cued(source_subtitle):
    subs = pysubs2.load(source_subtitle)
    cues = []
    for ev in subs.events:
        cues.append({
            'start_time': ev.start,
            'end_time': ev.end,
            'text': ev.text.strip(),
        })
    return sorted(cues, key=lambda c: c['start_time'])


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


if __name__ == "__main__":
    en = make_cued(
        "./test_subs/True Detective - S01E01 - The Long Bright Dark Bluray-1080p.en.srt"
    )
    zh = make_cued(
        "./test_subs/True Detective - S01E01 - The Long Bright Dark Bluray-1080p.zh.srt"
    )

    paired_subs = pair_subtitles(en, zh)
    write_synced_subs(paired_subs, "synced_subs", "en", "zh")


