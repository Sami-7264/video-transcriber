#!/usr/bin/env python3
"""Video Transcriber — command-line entry point.

Run ``python transcribe.py --help`` for full usage.

Examples:
    python transcribe.py input.mp4 --model large-v3
    python transcribe.py meeting.mp4 --device auto --language en --output ./output
    python transcribe.py ./videos --recursive --skip-existing
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable even when invoked from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.cli import run  # noqa: E402  (import after sys.path tweak)

if __name__ == "__main__":
    run()
