"""Video Transcriber — a local, GPU-aware CLI built on Faster Whisper.

The package is organised into small, single-responsibility modules:

* :mod:`app.utils`         — logging, custom exceptions, formatting helpers, constants.
* :mod:`app.audio`         — FFmpeg detection, media probing and audio extraction.
* :mod:`app.transcription` — device resolution and the Faster Whisper wrapper.
* :mod:`app.subtitles`     — writers for txt / srt / vtt / json / md / csv outputs.
* :mod:`app.cli`           — the Typer command-line interface (the user entry point).
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
