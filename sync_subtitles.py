#!/usr/bin/env python3

import pysubs2


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
            'text': ev.text.strip(),
        })
    return sorted(cues, key=lambda c: c['start_time'])


def pair_subtitles(tl_cued, sl_cued):
    ''''''
    times = set()
    for cue_list in (tl_cued, sl_cued):
        for cue in cue_list:
            times.add(cue['start_time'])
            times.add(cue['end_time'])
    timeline = sorted(times)
    paired_subs = []
    for t0, t1 in zip(timeline, timeline[1:]):
        tl_text = next((c["text"] for c in tl_cued if c["start_time"] <= t0 < c['end_time']), "")
        sl_text = next((c["text"] for c in sl_cued if c["start_time"] <= t0 < c['end_time']), "")
        
        if not (tl_text or sl_text):
            continue
        
        last = paired_subs[-1] if paired_subs else None
        if last and last['tl_text'] == tl_text and last['sl_text'] == sl_text:
            last['end_time'] = t1
        else:
            paired_subs.append({
               "start_time": t0,
               "end_time": t1,
               "tl_text": tl_text,
               "sl_text": sl_text
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


