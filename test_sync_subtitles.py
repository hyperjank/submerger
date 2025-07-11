import importlib
import pytest
import submerger.align as llm_align
from submerger.sync import SubtitleCue


@pytest.fixture(autouse=True)
def _stub_client(monkeypatch):
    monkeypatch.setattr(llm_align, "get_client", lambda: (None, None))


# ensure module imports without side effects like loading test files


def test_importable():
    mod = importlib.import_module("submerger.sync")
    assert hasattr(mod, "make_cued")


def test_pair_basic():
    from submerger.sync import pair_subtitles

    tl = [
        SubtitleCue(0, 1000, "A"),
        SubtitleCue(1000, 2000, "B"),
    ]
    sl = [
        SubtitleCue(0, 1500, "a"),
        SubtitleCue(1500, 2000, "b"),
    ]

    expected = [
        {"start_time": 0, "end_time": 1000, "tl_text": "A", "sl_text": "a"},
        {"start_time": 1000, "end_time": 1500, "tl_text": "B", "sl_text": "a"},
        {"start_time": 1500, "end_time": 2000, "tl_text": "B", "sl_text": "b"},
    ]

    assert pair_subtitles(tl, sl) == expected


def test_align_invokes_llm(tmp_path, monkeypatch):
    from submerger import align

    def write_srt(cues, path):
        import pysubs2

        subs = pysubs2.SSAFile()
        for c in cues:
            subs.events.append(
                pysubs2.SSAEvent(start=c.start_time, end=c.end_time, text=c.text)
            )
        subs.save(path)

    tl_cues = [SubtitleCue(0, 1000, "A")]
    sl_cues = [SubtitleCue(0, 1000, "a")]
    tl_path = tmp_path / "tl.srt"
    sl_path = tmp_path / "sl.srt"
    write_srt(tl_cues, tl_path)
    write_srt(sl_cues, sl_path)

    cleanup_called = False

    def fake_cleanup(tl_sample, sl_sample):
        nonlocal cleanup_called
        cleanup_called = True
        return tl_sample, sl_sample

    monkeypatch.setattr(align, "call_llm_cleanup", fake_cleanup)

    select_calls = []

    def fake_select(sl_text, tl_block):
        select_calls.append(sl_text)
        return "sel"

    monkeypatch.setattr(align, "call_llm_select_target", fake_select)

    align.main(str(tl_path), str(sl_path), out=str(tmp_path / "out"))

    assert cleanup_called
    assert select_calls


def test_make_cued_skips_empty(tmp_path):
    from submerger.sync import make_cued

    sample = """1
00:00:00,000 --> 00:00:01,000
<i></i>

2
00:00:01,000 --> 00:00:02,000
<b>   </b>

3
00:00:02,000 --> 00:00:03,000
Real text
"""
    p = tmp_path / "a.srt"
    p.write_text(sample)

    cues = make_cued(str(p))

    assert cues == [SubtitleCue(2000, 3000, "Real text")]


def test_no_empty_segments_reach_pair(tmp_path):
    from submerger.sync import make_cued, pair_subtitles

    srt = """1
00:00:00,000 --> 00:00:01,000
<i></i>

2
00:00:02,000 --> 00:00:03,000
Hello
"""
    p = tmp_path / "t.srt"
    p.write_text(srt)
    tl = make_cued(str(p))
    sl = [SubtitleCue(2000, 3000, "hola")]

    assert all(c.text for c in tl)

    paired = pair_subtitles(tl, sl)
    assert paired == [
        {"start_time": 2000, "end_time": 3000, "tl_text": "Hello", "sl_text": "hola"}
    ]


def test_dedupe_cues():
    from submerger.sync import dedupe_cues

    cues = [
        SubtitleCue(0, 1000, "hi"),
        SubtitleCue(1000, 2000, "hi"),
        SubtitleCue(2000, 3000, "there"),
        SubtitleCue(3000, 4000, "there"),
    ]

    deduped = dedupe_cues(cues)
    assert deduped == [
        SubtitleCue(0, 2000, "hi"),
        SubtitleCue(2000, 4000, "there"),
    ]


def test_write_scripts_dedupes(tmp_path):
    from submerger.sync import write_scripts

    collapsed = [
        {"start_time": 0, "end_time": 1000, "tl_text": "hi", "sl_text": "hola"},
        {"start_time": 1000, "end_time": 2000, "tl_text": "hi", "sl_text": "adios"},
        {"start_time": 2000, "end_time": 3000, "tl_text": "bye", "sl_text": "adios"},
        {"start_time": 3000, "end_time": 4000, "tl_text": "bye", "sl_text": "adios"},
    ]

    out = tmp_path / "script"
    write_scripts(collapsed, str(out), "en", "es")

    assert (tmp_path / "script_en.txt").read_text().splitlines() == ["hi", "bye"]
    assert (tmp_path / "script_es.txt").read_text().splitlines() == ["hola", "adios"]


def test_write_synced_subs_dedupes(tmp_path):
    from submerger.sync import write_synced_subs
    import pysubs2

    collapsed = [
        {"start_time": 0, "end_time": 1000, "tl_text": "hi", "sl_text": "hola"},
        {"start_time": 1000, "end_time": 2000, "tl_text": "hi", "sl_text": "adios"},
        {"start_time": 2000, "end_time": 3000, "tl_text": "bye", "sl_text": "adios"},
        {"start_time": 3000, "end_time": 4000, "tl_text": "bye", "sl_text": "adios"},
    ]

    out = tmp_path / "sub"
    write_synced_subs(collapsed, str(out), "en", "es")

    tl_events = [(e.start, e.end, e.text) for e in pysubs2.load(str(out) + "_en.srt")]
    sl_events = [(e.start, e.end, e.text) for e in pysubs2.load(str(out) + "_es.srt")]

    assert tl_events == [(0, 2000, "hi"), (2000, 4000, "bye")]
    assert sl_events == [(0, 1000, "hola"), (1000, 4000, "adios")]
