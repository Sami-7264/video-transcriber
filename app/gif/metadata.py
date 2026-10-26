"""Metadata model for extracted GIF frames and the ``metadata.json`` writer.

Plain dataclasses keep the data easy to build, test and serialise. The JSON
layout matches the documented schema (filename, dimensions, frame_count,
duration_ms, fps, loop_count, then a per-frame list).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .utils import OutputError


@dataclass(slots=True)
class FrameMeta:
    """Metadata for a single extracted frame."""

    index: int               # 1-based position in the GIF
    filename: str            # output filename, e.g. "frame_000001.png"
    duration_ms: int         # how long this frame is displayed
    timestamp_ms: int        # cumulative start time of this frame in the GIF

    def to_dict(self) -> dict:
        """Return the JSON-serialisable representation of this frame."""
        return {
            "index": self.index,
            "filename": self.filename,
            "duration_ms": self.duration_ms,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass(slots=True)
class GifMetadata:
    """Aggregate metadata describing a GIF and all of its extracted frames."""

    filename: str
    width: int
    height: int
    frame_count: int
    duration_ms: int
    fps: float
    loop_count: int
    gif_version: str = "GIF89a"
    mode: str = "RGBA"
    has_transparency: bool = False
    frames: list[FrameMeta] = field(default_factory=list)

    @staticmethod
    def compute_fps(frame_count: int, duration_ms: int) -> float:
        """Return the average frames-per-second, guarding against zero duration.

        Args:
            frame_count: Total number of frames.
            duration_ms: Total animation duration in milliseconds.

        Returns:
            Average FPS rounded to two decimals, or ``0.0`` if duration is zero.
        """
        if duration_ms <= 0 or frame_count <= 0:
            return 0.0
        return round(frame_count / (duration_ms / 1000.0), 2)

    def to_dict(self) -> dict:
        """Return the full JSON-serialisable metadata document."""
        return {
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_ms": self.duration_ms,
            "fps": self.fps,
            "loop_count": self.loop_count,
            "gif_version": self.gif_version,
            "mode": self.mode,
            "has_transparency": self.has_transparency,
            "frames": [frame.to_dict() for frame in self.frames],
        }


def write_metadata(metadata: GifMetadata, path: Path) -> Path:
    """Write *metadata* to *path* as pretty-printed JSON.

    Args:
        metadata: The metadata document to serialise.
        path: Destination ``metadata.json`` path.

    Returns:
        The path written.

    Raises:
        OutputError: If the file cannot be written.
    """
    try:
        path.write_text(
            json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise OutputError(f"Could not write metadata to '{path}': {exc}") from exc
    return path
