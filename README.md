## Bilingual Subtitle Player

A Linux video player with synchronized dual-language subtitles and AI-powered alignment/annotation, dictionary lookups for clicked words, AI explanation of highlighted phrases, and a plugin system for learning tools that display on video pause.

Features

* Parse and merge SRT/ASS files into unified timeline

* AI-driven semantic alignment between subtitles in two languages
* Regex cleanup of ads/URLs in the first and last 30 seconds
* Export perfectly synced L1 + L2 SRTs

* mpv playback with interactable subtitle display and custom placement
* Qt6/mpv player with a clean design
* Wayland support via libmpv video embedding

### Installation

Install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

To play a video alongside two subtitle tracks use ``qt_mpv_scroll.py``.  You can
either pass the files on the command line or open them via the ``File`` menu::

```bash
python qt_mpv_scroll.py movie.mp4 subtitles_en.srt subtitles_es.srt
```

When launched without arguments the player will prompt you to choose the video
and subtitle files.  The ``File`` menu also contains an ``Align Subtitles``
action which uses the built-in alignment logic to synchronize the currently
loaded tracks.
The player now includes basic playback controls. Use the on-screen play/pause
button or press the space bar to toggle playback. The slider lets you seek to a
specific position.

### Configuration

Project settings live in ``submerger/settings.json`` in the repository root. The ``llm``
section controls the language model credentials:

* ``api_key`` – API token for the language model service.
* ``api_base`` – Base URL of the API (defaults to ``http://localhost:1234/v1``).
* ``model`` – Model name to use (defaults to ``qwen3-8b``).

See [SETTINGS.md](SETTINGS.md) for additional settings.

You can then run ``python -m submerger.cli pair`` or ``python -m submerger.cli align`` with your two subtitle
files regardless of language. Both commands provide options for the
input files and language codes.
Use the ``--deepseek`` flag if you want to send requests to the public
DeepSeek service instead of the local API.


### Future:

* on-demand slang, cultural notes, grammar analysis via LLM
* plugin system for additional features
