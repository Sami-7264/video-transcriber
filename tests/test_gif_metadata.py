"""Unit tests for :mod:`app.gif.metadata` (pure logic, no Pillow)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.gif.metadata import FrameMeta, GifMetadata, write_metadata  # noqa: E402


def _sample_metadata() -> GifMetadata:
    frames = [
        FrameMeta(index=1, filename="frame_000001.png", duration_ms=33, timestamp_ms=0),
        FrameMeta(index=2, filename="frame_000002.png", duration_ms=33, timestamp_ms=33),
    ]
    return GifMetadata(
        filename="animation.gif",
        width=640,
        height=360,
        frame_count=2,
        duration_ms=66,
        fps=GifMetadata.compute_fps(2, 66),
        loop_count=0,
        gif_version="GIF89a",
        mode="P",
        has_transparency=True,
        frames=frames,
    )


def test_compute_fps_typical() -> None:
    assert GifMetadata.compute_fps(150, 5000) == 30.0


def test_compute_fps_zero_duration_is_safe() -> None:
    assert GifMetadata.compute_fps(10, 0) == 0.0
    assert GifMetadata.compute_fps(0, 1000) == 0.0


def test_frame_meta_to_dict() -> None:
    frame = FrameMeta(index=1, filename="frame_000001.png", duration_ms=33, timestamp_ms=0)
    assert frame.to_dict() == {
        "index": 1,
        "filename": "frame_000001.png",
        "duration_ms": 33,
        "timestamp_ms": 0,
    }


def test_metadata_to_dict_shape() -> None:
    data = _sample_metadata().to_dict()
    assert list(data)[:7] == [
        "filename", "width", "height", "frame_count", "duration_ms", "fps", "loop_count",
    ]
    assert data["width"] == 640
    assert data["frame_count"] == 2
    assert len(data["frames"]) == 2
    assert data["frames"][1]["timestamp_ms"] == 33


def test_write_metadata_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = write_metadata(_sample_metadata(), Path(tmp) / "metadata.json")
        loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["filename"] == "animation.gif"
    assert loaded["has_transparency"] is True


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
