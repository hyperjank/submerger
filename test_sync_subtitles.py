import importlib

# ensure module imports without side effects like loading test files

def test_importable():
    mod = importlib.import_module('sync_subtitles')
    assert hasattr(mod, 'make_cued')


def test_pair_basic():
    from sync_subtitles import pair_subtitles

    tl = [
        {'start_time': 0, 'end_time': 1000, 'text': 'A'},
        {'start_time': 1000, 'end_time': 2000, 'text': 'B'},
    ]
    sl = [
        {'start_time': 0, 'end_time': 1500, 'text': 'a'},
        {'start_time': 1500, 'end_time': 2000, 'text': 'b'},
    ]

    expected = [
        {'start_time': 0, 'end_time': 1000, 'tl_text': 'A', 'sl_text': 'a'},
        {'start_time': 1000, 'end_time': 1500, 'tl_text': 'B', 'sl_text': 'a'},
        {'start_time': 1500, 'end_time': 2000, 'tl_text': 'B', 'sl_text': 'b'},
    ]

    assert pair_subtitles(tl, sl) == expected



def test_needs_llm_heuristic():
    from llm_align import needs_llm

    good_tl = {'start_time': 0, 'end_time': 1000, 'text': 'Hello'}
    good_sl = {'start_time': 0, 'end_time': 1000, 'text': 'hello'}
    assert not needs_llm(good_tl, good_sl)

    bad_tl = {'start_time': 0, 'end_time': 1000, 'text': '你好'}
    bad_sl = {'start_time': 5000, 'end_time': 6000, 'text': 'world'}
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

    assert cues == [
        {'start_time': 2000, 'end_time': 3000, 'text': 'Real text'}
    ]


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
    sl = [{'start_time': 2000, 'end_time': 3000, 'text': 'hola'}]

    assert all(c['text'] for c in tl)

    paired = pair_subtitles(tl, sl)
    assert paired == [
        {'start_time': 2000, 'end_time': 3000, 'tl_text': 'Hello', 'sl_text': 'hola'}
    ]


def test_dedupe_cues():
    from sync_subtitles import dedupe_cues

    cues = [
        {'start_time': 0, 'end_time': 1000, 'text': 'hi'},
        {'start_time': 1000, 'end_time': 2000, 'text': 'hi'},
        {'start_time': 2000, 'end_time': 3000, 'text': 'there'},
        {'start_time': 3000, 'end_time': 4000, 'text': 'there'},
    ]

    deduped = dedupe_cues(cues)
    assert deduped == [
        {'start_time': 0, 'end_time': 1000, 'text': 'hi'},
        {'start_time': 2000, 'end_time': 3000, 'text': 'there'},
    ]

