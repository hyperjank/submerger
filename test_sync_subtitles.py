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
