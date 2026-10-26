"""Unit tests for :mod:`app.audio` that don't require FFmpeg to be installed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:  # standalone fallback: minimal pytest.raises shim
    import types

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, *_):
            if exc_type is None or not issubclass(exc_type, self.exc):
                raise AssertionError(f"expected {self.exc.__name__} to be raised")
            return True  # suppress the expected exception

    pytest = types.SimpleNamespace(raises=_Raises)  # type: ignore

import tempfile  # noqa: E402

from app.audio import SilenceSpan, validate_input_file  # noqa: E402
from app.utils import InvalidInputError  # noqa: E402


def test_validate_input_missing_file() -> None:
    with pytest.raises(InvalidInputError):
        validate_input_file(Path("/no/such/file.mp4"))


def test_validate_input_unsupported_format() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bogus = Path(tmp) / "notes.txt"
        bogus.write_text("hello", encoding="utf-8")
        with pytest.raises(InvalidInputError):
            validate_input_file(bogus)


def test_validate_input_accepts_supported_extension() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        media = Path(tmp) / "clip.mp4"
        media.write_bytes(b"\x00\x00")  # contents irrelevant for path validation
        assert validate_input_file(media) == media.resolve()


def test_silence_span_duration() -> None:
    assert SilenceSpan(start=1.0, end=3.5).duration == 2.5
    assert SilenceSpan(start=5.0, end=4.0).duration == 0.0


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
