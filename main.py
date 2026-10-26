#!/usr/bin/env python3
"""Unified command-line interface for the project.

Aggregates every feature under a single multi-command entry point:

    python main.py transcribe      meeting.mp4 --model large-v3
    python main.py extract-frames  animation.gif --format png

Each sub-command is the very same implementation used by the dedicated scripts
(``transcribe.py`` and ``extract_frames.py``); this module only wires them
together, so the existing tools keep working unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable even when invoked from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import typer  # noqa: E402

from app.cli import main as _transcribe_command  # noqa: E402
from app.gif.cli import contact_sheet as _contact_sheet_command  # noqa: E402
from app.gif.cli import extract as _extract_frames_command  # noqa: E402

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Local media toolkit — transcribe audio/video, extract GIF frames, build contact sheets.",
)

# Reuse the existing command callables as sub-commands (no duplication).
app.command("transcribe", help="Transcribe a video/audio file (or folder).")(
    _transcribe_command
)
app.command("extract-frames", help="Extract every frame of an animated GIF.")(
    _extract_frames_command
)
app.command("contact-sheet", help="Combine extracted frames into one contact sheet.")(
    _contact_sheet_command
)


def run() -> None:
    """Console entry point for the unified CLI."""
    app()


if __name__ == "__main__":
    run()
