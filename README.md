## Bilingual Subtitle Player

A Linux video player with synchronized dual-language subtitles and AI-powered alignment/annotation.

Features

* Parse and merge SRT/ASS files into unified timeline

* AI-driven chunk alignment with custom timestamps

* Export perfectly synced L1 + L2 SRTs

* GStreamer/mpv playback with top/bottom rendering
* Qt6/mpv player with scrolling subtitle columns

### Installation

Install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

To play a video alongside two subtitle tracks use ``qt_mpv_scroll.py``::

```bash
python qt_mpv_scroll.py movie.mp4 subtitles_en.srt subtitles_es.srt
```

### Configuration

The alignment scripts look for the following environment variables which can be
set in a ``.env`` file or your shell environment:

* ``LLM_API_KEY`` – API token for the language model service.
* ``LLM_API_BASE`` – Base URL of the API (e.g. ``http://localhost:8000/v1`` for
  a local model).  Defaults to the local endpoint.
* ``LLM_MODEL`` – Model name to use (defaults to ``deepseek-chat``).

You can then run ``sync_subtitles.py`` or ``llm_align.py`` with your two subtitle
files regardless of language. Both scripts provide command line options for the
input files and language codes.
Use the ``--deepseek`` flag if you want to send requests to the public
DeepSeek service instead of the local API.

The LLM is only consulted to answer simple yes/no questions about whether two
lines match and to optionally suggest removing intro/outro advertisements and
translator signatures. This keeps API calls minimal and fast.

### Future:

* on-demand slang, cultural notes, grammar analysis via LLM
