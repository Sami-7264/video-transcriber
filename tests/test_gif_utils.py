"""Unit tests for :mod:`app.gif.utils` (no Pillow required)."""

from __future__ import annotations

import sys
import tempfile
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
            return True

    pytest = types.SimpleNamespace(raises=_Raises)  # type: ignore

from app.gif.utils import (  # noqa: E402
    GifNotFoundError,
    InvalidGifError,
    OutputError,
    UnsupportedOutputFormatError,
    build_frame_filename,
    prepare_output_dir,
    read_gif_version,
    resolve_output_format,
    validate_gif_path,
)

# Minimal valid 1x1 GIF89a (used to exercise signature reading).
_TINY_GIF89A = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00"
    b"!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
)


def test_build_frame_filename_padding() -> None:
    assert build_frame_filename("frame_", 1, ".png") == "frame_000001.png"
    assert build_frame_filename("frame_", 150, ".png") == "frame_000150.png"
    assert build_frame_filename("img-", 42, ".webp") == "img-000042.webp"


def test_resolve_output_format() -> None:
    assert resolve_output_format("png") == ("png", ".png")
    assert resolve_output_format("JPG") == ("jpg", ".jpg")
    assert resolve_output_format("jpeg") == ("jpeg", ".jpg")
    assert resolve_output_format("webp") == ("webp", ".webp")


def test_resolve_output_format_invalid() -> None:
    with pytest.raises(UnsupportedOutputFormatError):
        resolve_output_format("bmp")


def test_validate_gif_path_missing() -> None:
    with pytest.raises(GifNotFoundError):
        validate_gif_path(Path("/no/such/file.gif"))


def test_validate_gif_path_wrong_extension() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bogus = Path(tmp) / "not.png"
        bogus.write_bytes(b"x")
        with pytest.raises(InvalidGifError):
            validate_gif_path(bogus)


def test_read_gif_version_valid() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "a.gif"
        gif.write_bytes(_TINY_GIF89A)
        assert read_gif_version(gif) == "GIF89a"


def test_read_gif_version_invalid_signature() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "a.gif"
        gif.write_bytes(b"NOTAGIF_payload")
        with pytest.raises(InvalidGifError):
            read_gif_version(gif)


def test_prepare_output_dir_creates_subfolder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = prepare_output_dir(Path(tmp), "My Animation", ".png", overwrite=False)
        assert out.exists()
        assert out.name == "my-animation"


def test_prepare_output_dir_refuses_existing_without_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = prepare_output_dir(Path(tmp), "clip", ".png", overwrite=False)
        (out / "frame_000001.png").write_bytes(b"x")
        with pytest.raises(OutputError):
            prepare_output_dir(Path(tmp), "clip", ".png", overwrite=False)


def test_prepare_output_dir_overwrite_clears_frames() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = prepare_output_dir(Path(tmp), "clip", ".png", overwrite=False)
        stale = out / "frame_000001.png"
        stale.write_bytes(b"x")
        prepare_output_dir(Path(tmp), "clip", ".png", overwrite=True)
        assert not stale.exists()


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
