"""Unit tests for :mod:`app.subtitles` writers."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.subtitles import (  # noqa: E402
    DEFAULT_FORMATS,
    expected_output_path,
    write_csv,
    write_outputs,
    write_srt,
    write_txt,
    write_vtt,
)
from factories import make_result  # noqa: E402


def test_write_txt() -> None:
    result = make_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_txt(result, Path(tmp) / "out.txt")
        content = path.read_text(encoding="utf-8")
    assert "Hello world." in content
    assert "This is a test." in content


def test_write_srt_format() -> None:
    result = make_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_srt(result, Path(tmp) / "out.srt")
        content = path.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello world." in content
    assert "2\n00:00:02,500 --> 00:00:05,000\nThis is a test." in content


def test_write_vtt_header_and_dot_marker() -> None:
    result = make_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_vtt(result, Path(tmp) / "out.vtt")
        content = path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in content


def test_write_json_roundtrip() -> None:
    result = make_result()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        write_outputs(result, Path(tmp), "out", formats=("json",))
        data = json.loads(out.read_text(encoding="utf-8"))
    assert data["metadata"]["model"] == "base"
    assert data["segments"][0]["confidence"] is not None
    assert data["segments"][0]["start"] == 0.0


def test_write_csv_header_and_rows() -> None:
    result = make_result()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_csv(result, Path(tmp) / "out.csv")
        lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "index,start,end,duration,confidence,text"
    assert len(lines) == 3  # header + 2 segments


def test_write_outputs_creates_all_default_formats() -> None:
    result = make_result()
    with tempfile.TemporaryDirectory() as tmp:
        written = write_outputs(result, Path(tmp), "video", DEFAULT_FORMATS)
        suffixes = {p.suffix for p in written}
        assert suffixes == {".txt", ".srt", ".vtt", ".json"}
        assert all(p.exists() for p in written)


def test_expected_output_path() -> None:
    path = expected_output_path(Path("/out"), "clip", "json")
    assert path == Path("/out/clip.json")


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
