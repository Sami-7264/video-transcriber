"""Unit tests for :mod:`app.features` (search + diarization assignment).

These avoid the heavy optional dependencies: they test the pure-Python logic
(speaker overlap assignment and JSON search) directly.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features import SpeakerTurn, assign_speakers, search_transcripts  # noqa: E402
from factories import make_result  # noqa: E402


def test_assign_speakers_by_overlap() -> None:
    result = make_result()  # seg0: 0–2.5, seg1: 2.5–5.0
    turns = [
        SpeakerTurn(start=0.0, end=2.4, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.5, end=5.0, speaker="SPEAKER_01"),
    ]
    assign_speakers(result, turns)
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[1].speaker == "SPEAKER_01"


def test_assign_speakers_no_overlap_leaves_none() -> None:
    result = make_result()
    assign_speakers(result, [SpeakerTurn(start=100.0, end=200.0, speaker="X")])
    assert result.segments[0].speaker is None


def test_search_finds_match_in_json() -> None:
    result = make_result()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "clip.json"
        target.write_text(json.dumps(result.to_dict()), encoding="utf-8")

        hits = search_transcripts(Path(tmp), "test")
        assert len(hits) == 1
        assert "test" in hits[0].text.lower()
        assert hits[0].start == 2.5


def test_search_is_case_insensitive_by_default() -> None:
    result = make_result()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "clip.json"
        target.write_text(json.dumps(result.to_dict()), encoding="utf-8")
        assert search_transcripts(Path(tmp), "HELLO")
        assert not search_transcripts(Path(tmp), "HELLO", case_sensitive=True)


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
