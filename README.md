## Bilingual Subtitle Player

A Linux video player with synchronized dual-language subtitles and AI-powered alignment/annotation.

Features

* Parse and merge SRT/ASS files into unified timeline

* AI-driven chunk alignment with custom timestamps

* Export perfectly synced L1 + L2 SRTs

* GStreamer/mpv playback with top/bottom rendering

### Installation

Install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

### Configuration

The alignment scripts look for the following environment variables which can be
set in a ``.env`` file or your shell environment:

* ``LLM_API_KEY`` – API token for the language model service.
* ``LLM_API_BASE`` – Base URL of the API (e.g. ``http://localhost:8000/v1`` for
  a local model).  Defaults to the public DeepSeek endpoint.
* ``LLM_MODEL`` – Model name to use (defaults to ``deepseek-chat``).

You can then run ``sync_subtitles.py`` or ``llm_align.py`` with your two subtitle
files regardless of language. Both scripts provide command line options for the
input files and language codes.

### Future:

* on-demand slang, cultural notes, grammar analysis via LLM
