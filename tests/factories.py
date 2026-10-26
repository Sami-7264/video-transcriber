"""Shared test helpers for building sample transcription results.

Kept fixture-free (plain functions) so the test modules can run both under
pytest and as standalone scripts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.transcription import Segment, TranscriptionResult, Word  # noqa: E402


def make_result() -> TranscriptionResult:
    """Return a small, deterministic :class:`TranscriptionResult` for tests."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=2.5,
            text="Hello world.",
            avg_logprob=math.log(0.9),  # -> confidence ~0.9
            no_speech_prob=0.01,
            compression_ratio=1.2,
            words=[
                Word(start=0.0, end=1.0, word="Hello", probability=0.99),
                Word(start=1.0, end=2.5, word="world.", probability=0.95),
            ],
        ),
        Segment(
            id=1,
            start=2.5,
            end=5.0,
            text="This is a test.",
            avg_logprob=math.log(0.8),
            no_speech_prob=0.02,
            compression_ratio=1.1,
        ),
    ]
    return TranscriptionResult(
        segments=segments,
        language="en",
        language_probability=0.98,
        audio_duration=5.0,
        model="base",
        device="cpu",
        compute_type="int8",
        source_file="/tmp/sample.mp4",
        elapsed_seconds=2.0,
    )


def run_module_tests(namespace: dict) -> int:
    """Run every ``test_*`` callable in *namespace*; return the failure count.

    Lets each test file double as a standalone script when pytest is absent.
    """
    failures = 0
    for name, obj in sorted(namespace.items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{'OK' if not failures else 'FAILED'} — {failures} failure(s)")
    return failures
