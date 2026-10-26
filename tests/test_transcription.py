"""Unit tests for :mod:`app.transcription` (no model download required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.transcription import Segment, resolve_device  # noqa: E402
from factories import make_result  # noqa: E402


def test_segment_confidence_from_logprob() -> None:
    import math

    seg = Segment(id=0, start=0, end=1, text="x", avg_logprob=math.log(0.5))
    assert seg.confidence == 0.5


def test_segment_confidence_none_when_missing() -> None:
    seg = Segment(id=0, start=0, end=1, text="x", avg_logprob=None)
    assert seg.confidence is None


def test_segment_to_dict_keys() -> None:
    seg = Segment(id=0, start=0.1234, end=1.0, text=" hi ", avg_logprob=-0.1)
    data = seg.to_dict()
    assert data["start"] == 0.123
    assert set(data) >= {"id", "start", "end", "text", "speaker", "confidence", "words"}


def test_result_text_property() -> None:
    result = make_result()
    assert result.text == "Hello world. This is a test."


def test_result_speed_property() -> None:
    result = make_result()  # 5s audio / 2s elapsed
    assert result.speed == 2.5


def test_result_to_dict_structure() -> None:
    data = make_result().to_dict()
    assert data["metadata"]["language"] == "en"
    assert data["metadata"]["speed_realtime_factor"] == 2.5
    assert len(data["segments"]) == 2
    assert data["segments"][0]["words"][0]["word"] == "Hello"


def test_resolve_device_cpu_default_compute() -> None:
    device, compute = resolve_device("cpu")
    assert device == "cpu"
    assert compute == "int8"


def test_resolve_device_explicit_compute_respected() -> None:
    device, compute = resolve_device("cpu", "float32")
    assert (device, compute) == ("cpu", "float32")


def test_resolve_device_auto_returns_valid() -> None:
    device, compute = resolve_device("auto")
    assert device in {"cpu", "cuda"}
    assert compute in {"int8", "float16"}


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
