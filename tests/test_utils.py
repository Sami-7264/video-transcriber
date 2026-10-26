"""Unit tests for :mod:`app.utils`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils import (  # noqa: E402
    format_duration,
    format_timestamp,
    is_media_file,
    is_video_file,
    slugify,
)


def test_format_timestamp_srt_uses_comma() -> None:
    assert format_timestamp(3661.5, decimal_marker=",") == "01:01:01,500"


def test_format_timestamp_vtt_uses_dot() -> None:
    assert format_timestamp(62.25) == "00:01:02,250".replace(",", ".")


def test_format_timestamp_zero_and_negative() -> None:
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(-5.0) == "00:00:00.000"


def test_format_timestamp_rounds_milliseconds() -> None:
    assert format_timestamp(1.2345) == "00:00:01.234" or format_timestamp(1.2345) == "00:00:01.235"


def test_format_duration_seconds() -> None:
    assert format_duration(45.234) == "45.2s"


def test_format_duration_minutes() -> None:
    assert format_duration(125) == "2m 05s"


def test_format_duration_hours() -> None:
    assert format_duration(3725) == "1h 02m 05s"


def test_slugify() -> None:
    assert slugify("My Cool Video!.mp4") == "my-cool-video.mp4"
    assert slugify("   ") == "transcript"


def test_is_media_file() -> None:
    assert is_media_file(Path("a.mp4"))
    assert is_media_file(Path("a.MP3"))
    assert not is_media_file(Path("a.txt"))


def test_is_video_file() -> None:
    assert is_video_file(Path("a.mkv"))
    assert not is_video_file(Path("a.wav"))


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
