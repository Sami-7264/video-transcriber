#!/usr/bin/env python3
"""GIF frame extraction — standalone command-line entry point.

Run ``python extract_frames.py --help`` for full usage.

Examples:
    python extract_frames.py animation.gif
    python extract_frames.py animation.gif --output ./frames --format png --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable even when invoked from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.gif.cli import run  # noqa: E402  (import after sys.path tweak)

if __name__ == "__main__":
    run()
