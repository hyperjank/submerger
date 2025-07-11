import importlib
import pytest
import llm_align
from sync_subtitles import SubtitleCue


@pytest.fixture(autouse=True)
def _stub_client(monkeypatch):
    monkeypatch.setattr(llm_align, "get_client", lambda: None)

# ensure module imports without side effects like loading test files

def test_importable():
    mod = importlib.import_module('sync_subtitles')
    assert hasattr(mod, 'make_cued')


def test_pair_basic():
    from sync_subtitles import pair_subtitles

    tl = [
        SubtitleCue(0, 1000, 'A'),
        SubtitleCue(1000, 2000, 'B'),
    ]
    sl = [
        SubtitleCue(0, 1500, 'a'),
        SubtitleCue(1500, 2000, 'b'),
    ]

    expected = [
        {'start_time': 0, 'end_time': 1000, 'tl_text': 'A', 'sl_text': 'a'},
        {'start_time': 1000, 'end_time': 1500, 'tl_text': 'B', 'sl_text': 'a'},
        {'start_time': 1500, 'end_time': 2000, 'tl_text': 'B', 'sl_text': 'b'},
    ]

    assert pair_subtitles(tl, sl) == expected



def test_needs_llm_heuristic():
    from llm_align import needs_llm

    good_tl = SubtitleCue(0, 1000, 'Hello')
    good_sl = SubtitleCue(0, 1000, 'hello')
    assert not needs_llm(good_tl, good_sl)

    bad_tl = SubtitleCue(0, 1000, '你好')
    bad_sl = SubtitleCue(5000, 6000, 'world')
    assert needs_llm(bad_tl, bad_sl)

def test_make_cued_skips_empty(tmp_path):
    from sync_subtitles import make_cued

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

    assert cues == [SubtitleCue(2000, 3000, 'Real text')]


def test_no_empty_segments_reach_pair(tmp_path):
    from sync_subtitles import make_cued, pair_subtitles

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
    sl = [SubtitleCue(2000, 3000, 'hola')]

    assert all(c.text for c in tl)

    paired = pair_subtitles(tl, sl)
    assert paired == [
        {'start_time': 2000, 'end_time': 3000, 'tl_text': 'Hello', 'sl_text': 'hola'}
    ]


def test_dedupe_cues():
    from sync_subtitles import dedupe_cues

    cues = [
        SubtitleCue(0, 1000, 'hi'),
        SubtitleCue(1000, 2000, 'hi'),
        SubtitleCue(2000, 3000, 'there'),
        SubtitleCue(3000, 4000, 'there'),
    ]

    deduped = dedupe_cues(cues)
    assert deduped == [
        SubtitleCue(0, 2000, 'hi'),
        SubtitleCue(2000, 4000, 'there'),
    ]


def test_write_scripts_dedupes(tmp_path):
    from sync_subtitles import write_scripts

    collapsed = [
        {'start_time': 0, 'end_time': 1000, 'tl_text': 'hi', 'sl_text': 'hola'},
        {'start_time': 1000, 'end_time': 2000, 'tl_text': 'hi', 'sl_text': 'adios'},
        {'start_time': 2000, 'end_time': 3000, 'tl_text': 'bye', 'sl_text': 'adios'},
        {'start_time': 3000, 'end_time': 4000, 'tl_text': 'bye', 'sl_text': 'adios'},
    ]

    out = tmp_path / "script"
    write_scripts(collapsed, str(out), "en", "es")

    assert (tmp_path / "script_en.txt").read_text().splitlines() == ['hi', 'bye']
    assert (tmp_path / "script_es.txt").read_text().splitlines() == ['hola', 'adios']


def test_write_synced_subs_dedupes(tmp_path):
    from sync_subtitles import write_synced_subs
    import pysubs2

    collapsed = [
        {'start_time': 0, 'end_time': 1000, 'tl_text': 'hi', 'sl_text': 'hola'},
        {'start_time': 1000, 'end_time': 2000, 'tl_text': 'hi', 'sl_text': 'adios'},
        {'start_time': 2000, 'end_time': 3000, 'tl_text': 'bye', 'sl_text': 'adios'},
        {'start_time': 3000, 'end_time': 4000, 'tl_text': 'bye', 'sl_text': 'adios'},
    ]

    out = tmp_path / "sub"
    write_synced_subs(collapsed, str(out), "en", "es")

    tl_events = [(e.start, e.end, e.text) for e in pysubs2.load(str(out) + "_en.srt")] 
    sl_events = [(e.start, e.end, e.text) for e in pysubs2.load(str(out) + "_es.srt")]

    assert tl_events == [(0, 2000, 'hi'), (2000, 4000, 'bye')]
    assert sl_events == [(0, 1000, 'hola'), (1000, 4000, 'adios')]


def test_adjust_aligned_timings_merges_nearby():
    from sync_subtitles import pair_subtitles
    from llm_align import align_with_llm, adjust_aligned_timings

    tl = [SubtitleCue(0, 1000, 'hi')]
    sl = [SubtitleCue(50, 800, 'hola')]

    collapsed = pair_subtitles(tl, sl)
    aligned = align_with_llm(collapsed, time_tolerance=200)
    adjusted = adjust_aligned_timings(aligned, time_tolerance=200)

    assert adjusted == [
        {'start_time': 0, 'end_time': 1000, 'tl_text': 'hi', 'sl_text': 'hola'}
    ]


def test_regex_cleanup_removes_urls():
    from llm_align import regex_cleanup

    cues = [
        SubtitleCue(0, 1000, 'visit http://example.com'),
        SubtitleCue(35000, 36000, 'end'),
    ]

    cleaned = regex_cleanup(cues, window_ms=30000)

    assert cleaned == [
        SubtitleCue(0, 1000, 'visit'),
        SubtitleCue(35000, 36000, 'end'),
    ]


def test_semantic_align_cues_basic():
    from llm_align import semantic_align_cues

    tl = [SubtitleCue(0, 1000, 'hello')]
    sl = [SubtitleCue(50, 900, 'hello')]

    aligned = semantic_align_cues(tl, sl, window_ms=200)
    assert aligned == [
        {'start_time': 50, 'end_time': 900, 'tl_text': 'hello', 'sl_text': 'hello'}
    ]


