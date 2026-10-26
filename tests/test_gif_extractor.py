"""End-to-end unit tests for :mod:`app.gif.extractor`.

A small animated GIF is synthesised with Pillow, then extracted and verified.
If Pillow is unavailable the tests skip gracefully (under pytest) or no-op
(standalone), so the suite never hard-fails on a minimal environment.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:  # pragma: no cover - standalone fallback
    pytest = None  # type: ignore

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_PIL = False

from app.gif.extractor import FrameContext, extract_frames  # noqa: E402
from app.gif.utils import InvalidGifError, OutputError  # noqa: E402

FRAME_COUNT = 6
FRAME_SIZE = (12, 8)  # (width, height)
FRAME_DURATION_MS = 100
_COLORS = [
    (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255),
    (255, 255, 0, 255), (255, 0, 255, 255), (0, 255, 255, 255),
]


def _skip_if_no_pil() -> bool:
    """Return True (and skip under pytest) when Pillow is missing."""
    if HAS_PIL:
        return False
    if pytest is not None:
        pytest.skip("Pillow not installed")
    print("  (skipped: Pillow not installed)")
    return True


def _make_gif(path: Path) -> None:
    """Write a deterministic animated GIF89a with distinct per-frame colours."""
    frames = [Image.new("RGBA", FRAME_SIZE, _COLORS[i]) for i in range(FRAME_COUNT)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        disposal=2,
    )


def test_extract_basic_png() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "clip.gif"
        _make_gif(gif)
        result = extract_frames(gif, output_dir=Path(tmp) / "out")

        assert result.metadata.frame_count == FRAME_COUNT
        assert len(result.frame_paths) == FRAME_COUNT
        # All frames exist, correctly named and ordered.
        for i, fp in enumerate(result.frame_paths, start=1):
            assert fp.name == f"frame_{i:06d}.png"
            assert fp.exists()
        # Output sub-folder is named after the GIF stem.
        assert result.output_dir.name == "clip"


def test_extracted_frames_preserve_dimensions_and_alpha() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "clip.gif"
        _make_gif(gif)
        result = extract_frames(gif, output_dir=Path(tmp) / "out")
        with Image.open(result.frame_paths[0]) as frame:
            assert frame.size == FRAME_SIZE
            assert frame.mode == "RGBA"  # transparency channel preserved


def test_metadata_contents() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "clip.gif"
        _make_gif(gif)
        result = extract_frames(gif, output_dir=Path(tmp) / "out")
        data = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert data["filename"] == "clip.gif"
    assert data["width"] == FRAME_SIZE[0]
    assert data["height"] == FRAME_SIZE[1]
    assert data["frame_count"] == FRAME_COUNT
    assert data["duration_ms"] == FRAME_COUNT * FRAME_DURATION_MS
    assert data["fps"] == round(FRAME_COUNT / (FRAME_COUNT * FRAME_DURATION_MS / 1000), 2)
    assert data["gif_version"] in ("GIF87a", "GIF89a")
    # Per-frame timestamps accumulate correctly.
    assert data["frames"][0]["timestamp_ms"] == 0
    assert data["frames"][1]["timestamp_ms"] == FRAME_DURATION_MS


def test_overwrite_behaviour() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "clip.gif"
        _make_gif(gif)
        out = Path(tmp) / "out"
        extract_frames(gif, output_dir=out)

        # Second run without overwrite must refuse.
        raised = False
        try:
            extract_frames(gif, output_dir=out)
        except OutputError:
            raised = True
        assert raised

        # With overwrite it succeeds again.
        result = extract_frames(gif, output_dir=out, overwrite=True)
        assert result.metadata.frame_count == FRAME_COUNT


def test_webp_and_jpg_formats() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "clip.gif"
        _make_gif(gif)
        webp = extract_frames(gif, output_dir=Path(tmp) / "w", image_format="webp")
        assert webp.frame_paths[0].suffix == ".webp"
        jpg = extract_frames(gif, output_dir=Path(tmp) / "j", image_format="jpg")
        assert jpg.frame_paths[0].suffix == ".jpg"


def test_no_metadata_flag() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "clip.gif"
        _make_gif(gif)
        result = extract_frames(gif, output_dir=Path(tmp) / "out", write_metadata_file=False)
        assert result.metadata_path is None
        assert not (result.output_dir / "metadata.json").exists()


def test_frame_processor_hook_receives_all_frames() -> None:
    if _skip_if_no_pil():
        return

    class _Collector:
        def __init__(self) -> None:
            self.indices: list[int] = []

        def process(self, context: FrameContext) -> None:
            self.indices.append(context.index)

    collector = _Collector()
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "clip.gif"
        _make_gif(gif)
        extract_frames(gif, output_dir=Path(tmp) / "out", processors=[collector])
    assert collector.indices == list(range(1, FRAME_COUNT + 1))


def test_progress_callback_invoked() -> None:
    if _skip_if_no_pil():
        return
    seen: list[tuple[int, int]] = []
    with tempfile.TemporaryDirectory() as tmp:
        gif = Path(tmp) / "clip.gif"
        _make_gif(gif)
        extract_frames(
            gif, output_dir=Path(tmp) / "out",
            progress_callback=lambda c, t: seen.append((c, t)),
        )
    assert seen[0] == (1, FRAME_COUNT)
    assert seen[-1] == (FRAME_COUNT, FRAME_COUNT)


def test_invalid_gif_signature_rejected() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake.gif"
        fake.write_bytes(b"NOTAGIF" + b"\x00" * 32)
        raised = False
        try:
            extract_frames(fake, output_dir=Path(tmp) / "out")
        except InvalidGifError:
            raised = True
        assert raised


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
