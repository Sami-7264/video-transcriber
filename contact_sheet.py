#!/usr/bin/env python3
"""Contact-sheet generator — standalone command-line entry point.

Run ``python contact_sheet.py --help`` for full usage.

Examples:
    python contact_sheet.py output/animation
    python contact_sheet.py output/animation --columns 9 --padding 8 --labels --timestamps
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable even when invoked from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.gif.cli import run_contact_sheet  # noqa: E402  (import after sys.path tweak)

if __name__ == "__main__":
    run_contact_sheet()
