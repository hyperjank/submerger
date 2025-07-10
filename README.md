## Bilingual Subtitle Player

A Linux video player with synchronized dual-language subtitles and AI-powered alignment/annotation, dictionary lookups for clicked words, AI explanation of highlighted phrases, and a plugin system for learning tools that display on video pause.

Features

* Parse and merge SRT/ASS files into unified timeline

* AI-driven semantic alignment between subtitles in two languages

* Export perfectly synced L1 + L2 SRTs

* mpv playback with interactable subtitle display and custom placement
* Qt6/mpv player with a clean design

### Installation

Install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

To play a video alongside two subtitle tracks use ``qt_mpv_scroll.py``::

```bash
python qt_mpv_scroll.py movie.mp4 subtitles_en.srt subtitles_es.srt
```
The player now includes basic playback controls. Use the on-screen play/pause
button or press the space bar to toggle playback. The slider lets you seek to a
specific position.

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


### Future:

* on-demand slang, cultural notes, grammar analysis via LLM
* plugin system for additional features
