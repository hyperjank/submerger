# Submerger

Submerger is an MVP Qt video player backed by mpv with a custom subtitle
overlay. It disables mpv's built-in subtitles and renders two external SRT
tracks at the same time.

## Features

- mpv video playback embedded in a PySide6 window
- open video, primary subtitle, and secondary subtitle files independently
- custom SRT parser and clock-driven subtitle renderer
- simultaneous two-language subtitle display
- external subtitle autodetection and independent embedded subtitle selection
- line replay/navigation, speed, subtitle-delay, seek, and fullscreen controls
- drag-and-drop, recent episodes, and automatic session restoration

## Requirements

- Python 3.10+
- mpv and libmpv installed on the system
- Python dependencies from `pyproject.toml`

On Debian/Ubuntu-like systems, the system dependency is usually:

```bash
sudo apt install mpv libmpv-dev
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate  # bash/zsh
pip install -e .
submerger
```

For fish, activate with:

```fish
source .venv/bin/activate.fish
submerger
```

Activation is optional; from the project root you can always launch directly:

```bash
./.venv/bin/submerger
```

Open an episode directly from fish. Language-tagged sidecars such as
`episode.en.srt`, `episode.zh.srt`, and `episode.alignment.json` are detected
automatically:

```fish
./.venv/bin/submerger /path/to/episode.mkv
```

Override detection when needed:

```fish
./.venv/bin/submerger /path/to/episode.mkv \
    --primary /path/to/episode.en.srt \
    --secondary /path/to/episode.zh.srt
```

Use `--no-restore` to open an empty player instead of restoring the previous
episode.

You can also launch without installing the console script:

```bash
PYTHONPATH=src python3 -m submerger.main
```

## Development

Run the test suite headlessly from an activated project environment:

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

Large media samples, generated alignment outputs, local environments, and LLM
credentials are intentionally excluded from version control. Keep disposable
alignment runs under `outputs/` or another untracked working directory.

## Everyday Playback

Open one video instead of choosing three files. Submerger searches beside it for
matching, language-tagged SRT files and an alignment sidecar. You can also drop a
video, one or two SRT files, or an alignment sidecar onto the window. Text-based
embedded tracks are available independently under `Subtitles -> Primary
Embedded Track` and `Secondary Embedded Track`; image-based tracks such as PGS
cannot feed the interactive text overlay and are marked unavailable.

Playback position, selected files/tracks, speed, and both subtitle delays are
saved every few seconds and when the window closes. The last episode resumes on
the next launch, and the ten most recent episodes appear under `File -> Recent
Episodes`. State is stored at `~/.local/state/submerger/playback.json` by default.

| Action | Shortcut |
| --- | --- |
| Play or pause | `Space` |
| Previous / next subtitle | `Ctrl+Left` / `Ctrl+Right` |
| Replay current subtitle | `R` |
| Seek backward / forward 5 seconds | `Left` / `Right` |
| Slower / faster | `[` / `]` |
| Reset speed | `Backspace` |
| Primary subtitle earlier / later | `Z` / `X` |
| Secondary subtitle earlier / later | `Shift+Z` / `Shift+X` |
| Reset both subtitle delays | `Ctrl+0` |
| Fullscreen | `F11` or double-click the video |
| Leave fullscreen | `Escape` |

## Align Subtitles

The offline aligner cleans the primary subtitle file into fuller utterance
segments and asks an alignment provider to map secondary subtitle cue ids to
each primary segment. Its canonical output is a versioned `.alignment.json`
sidecar containing source provenance, validator findings, candidate cues, and
human review state. The CLI also exports:

- cleaned primary SRT
- secondary SRT retimed to primary segment timestamps
- JSON sidecar used directly by the player

Dry-run with deterministic time-overlap alignment:

```bash
submerger-align primary.en.srt secondary.zh.srt \
  --primary-language en \
  --secondary-language zh \
  --output-prefix episode
```

Use an OpenAI-compatible chat completions endpoint:

```bash
OPENAI_API_KEY=... submerger-align primary.en.srt secondary.zh.srt \
  --provider openai \
  --model gpt-4.1-mini \
  --primary-language en \
  --secondary-language zh \
  --output-prefix episode
```

The LLM prompt asks for secondary cue ids, not rewritten subtitle text. Submerger
then builds the aligned secondary subtitle locally from the selected source cues
and assigns the primary segment timestamps only when an SRT export is requested.

Alignment caches are reused only when both source-file hashes, languages, model,
endpoint, prompt/pipeline version, batching, and window settings still match.
Malformed model responses are repaired where safe and otherwise become explicit
review items; they cannot silently overwrite another segment.

The same pipeline is available in the player through the `Align` button. The
alignment dock can use the currently loaded subtitle paths, run heuristic or
OpenAI-compatible alignment in the background, and show per-batch progress. Its
`Review` tab lists unresolved segments, shows the primary text and nearby source
cues, and lets you approve the current mapping or check replacement cues. Reviews
are saved to the sidecar and can be loaded directly into the player. Retimed SRT
files are an optional export and are disabled by default in the GUI.

## Phrase Explanation

Drag-select subtitle text in the player to request a phrase explanation from an
OpenAI-compatible endpoint. Defaults target a local LM Studio server:

```bash
SUBMERGER_LLM_BASE_URL=http://192.168.86.113:1234/v1 \
SUBMERGER_LLM_MODEL=qwen3.5-4b \
.venv/bin/submerger
```

Use `Settings -> LLM Endpoint Settings` to switch between LM Studio, OpenAI,
DeepSeek, or a custom OpenAI-compatible endpoint while comparing models. The
same saved endpoint settings drive phrase explanations, sentence diagrams, and
the alignment dock defaults. Selecting a provider in either the settings dialog
or alignment dock fills its model, URL, API key, and timeout preset together;
select Custom to edit those values directly. `SUBMERGER_LLM_API_KEY` may be any non-empty value
for LM Studio unless your server is configured to require a specific key.

## Sentence Diagrams

The built-in sentence diagram plugin calls the configured LLM on demand for the
currently selected subtitle segment. The model returns a structured set of
language units, grammatical/semantic relations, and cross-language meaning
links. Submerger renders that structure as responsive HTML in the Language Tools
dock instead of using fixed-width ASCII tables, so long segments can wrap with
the panel width.

## Episode Script Sidebar

Use the `Script` button in the player controls to show or hide the dockable
episode script sidebar. The sidebar displays the loaded subtitles for the full
episode, highlights the active subtitle during playback, and can automatically
scroll with playback. Use the `Highlight` and `Auto-scroll` checkboxes in the
sidebar to disable either behavior while you manually browse the script.

## Plugins

Submerger has a small plugin API for subtitle-driven tools. Built-in plugins
currently include dictionary lookup, LLM phrase explanation, and LLM-backed
sentence diagramming. Drag-select subtitle text to open the Language Tools dock,
then use the plugin action buttons.

User plugins can be placed in:

```bash
~/.local/share/submerger/plugins
```

or another directory selected with `SUBMERGER_PLUGIN_DIR`. A plugin file should
export either `plugin` or `create_plugin()` and implement:

```python
from submerger.plugins.base import PluginAction, PluginResult

class MyPlugin:
    plugin_id = "my_plugin"
    name = "My Plugin"
    actions = (PluginAction(plugin_id, "Run My Plugin", ("selection",)),)

    def run(self, action, context):
        return PluginResult("My Plugin", context.primary_text, "my-plugin")

def create_plugin():
    return MyPlugin()
```

## MVP Scope

The subtitle engine currently supports SubRip (`.srt`) files. It strips common
SRT inline tags, preserves multi-line cues, and chooses the active cue for each
track based on mpv's current playback timestamp.
