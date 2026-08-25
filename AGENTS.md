# Submerger project instructions

Submerger is a Python/PySide6 frontend around libmpv for bilingual subtitle
playback and language-learning tools.

- Keep source under `src/submerger` and tests under `tests`.
- The user's interactive shell is fish. Use fish-compatible commands or bypass activation with explicit `.venv/bin/...` launchers; do not tell the user to source the bash `activate` script.
- Run `QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v` after changes.
- Do not commit virtual environments, downloaded media, generated alignment outputs, API keys, or local endpoint settings.
- Preserve original subtitle text and timing unless a feature explicitly produces an export; provenance belongs in the alignment sidecar.
- Prefer targeted improvements to the existing Qt/libmpv application over introducing another application framework.
