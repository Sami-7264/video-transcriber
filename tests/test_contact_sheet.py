"""Unit tests for :mod:`app.gif.contact_sheet`.

Pure grid/label logic runs without Pillow; the end-to-end generation tests
synthesise real frame PNGs with Pillow and skip gracefully if it is absent.
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

from app.gif.contact_sheet import (  # noqa: E402
    compute_grid,
    compute_layout,
    discover_frames,
    format_timestamp_label,
    generate_contact_sheet,
)
from app.gif.utils import ContactSheetError, NoFramesFoundError  # noqa: E402

FRAME_W, FRAME_H = 40, 30


def _skip_if_no_pil() -> bool:
    if HAS_PIL:
        return False
    if pytest is not None:
        pytest.skip("Pillow not installed")
    print("  (skipped: Pillow not installed)")
    return True


def _make_frames(directory: Path, count: int, *, with_metadata: bool = False) -> None:
    """Write *count* numbered frame PNGs (and optional metadata.json)."""
    frames_meta = []
    for i in range(1, count + 1):
        name = f"frame_{i:06d}.png"
        color = (i * 7 % 255, i * 13 % 255, i * 29 % 255, 255)
        Image.new("RGBA", (FRAME_W, FRAME_H), color).save(directory / name)
        frames_meta.append(
            {"index": i, "filename": name, "duration_ms": 100, "timestamp_ms": (i - 1) * 100}
        )
    if with_metadata:
        (directory / "metadata.json").write_text(
            json.dumps({"filename": "x.gif", "frames": frames_meta}), encoding="utf-8"
        )


# --- Pure logic (no Pillow) -------------------------------------------------
def test_compute_grid_square_counts() -> None:
    assert compute_grid(16) == (4, 4)
    assert compute_grid(25) == (5, 5)
    assert compute_grid(36) == (6, 6)
    assert compute_grid(81) == (9, 9)
    assert compute_grid(100) == (10, 10)


def test_compute_grid_non_square() -> None:
    assert compute_grid(150) == (13, 12)  # ceil(sqrt(150))=13, ceil(150/13)=12
    assert compute_grid(10) == (4, 3)
    assert compute_grid(1) == (1, 1)


def test_compute_grid_explicit_columns() -> None:
    assert compute_grid(81, columns=9) == (9, 9)
    assert compute_grid(80, columns=10) == (10, 8)
    # More columns than frames is capped to the frame count.
    assert compute_grid(5, columns=20) == (5, 1)


def test_compute_grid_invalid() -> None:
    err = ContactSheetError
    for call in (lambda: compute_grid(0), lambda: compute_grid(10, columns=0)):
        raised = False
        try:
            call()
        except err:
            raised = True
        assert raised


def test_compute_layout_dimensions() -> None:
    layout = compute_layout(81, 180, 135, columns=9, padding=8)
    assert layout["columns"] == 9 and layout["rows"] == 9
    # width = 9*180 + 10*8 ; height = 9*135 + 10*8
    assert layout["width"] == 9 * 180 + 10 * 8
    assert layout["height"] == 9 * 135 + 10 * 8


def test_format_timestamp_label() -> None:
    assert format_timestamp_label(0) == "00:00.00"
    assert format_timestamp_label(130) == "00:00.13"
    assert format_timestamp_label(270) == "00:00.27"
    assert format_timestamp_label(65_000) == "01:05.00"


# --- Frame discovery --------------------------------------------------------
def test_discover_frames_sorted_without_metadata() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 3)
        refs = discover_frames(d)
        assert [r.path.name for r in refs] == [
            "frame_000001.png", "frame_000002.png", "frame_000003.png"
        ]
        assert all(r.timestamp_ms is None for r in refs)


def test_discover_frames_uses_metadata_order_and_timestamps() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 3, with_metadata=True)
        refs = discover_frames(d)
        assert [r.timestamp_ms for r in refs] == [0, 100, 200]


def test_discover_frames_excludes_existing_sheet() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 2)
        Image.new("RGBA", (10, 10)).save(d / "contact_sheet.png")
        refs = discover_frames(d)
        assert all("contact_sheet" not in r.path.name for r in refs)
        assert len(refs) == 2


# --- End-to-end generation --------------------------------------------------
def test_generate_basic_sheet() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 9)
        result = generate_contact_sheet(d, thumbnail_width=40, padding=8)
        assert (result.columns, result.rows) == (3, 3)
        assert result.output_path.exists()
        assert result.output_path.name == "contact_sheet.png"
        with Image.open(result.output_path) as sheet:
            # 3 cols * 40 + 4 padding gaps = 120 + 32 = 152 ; cell_h = 40*30/40 = 30
            assert sheet.size == (3 * 40 + 4 * 8, 3 * 30 + 4 * 8)
            assert sheet.mode == "RGBA"


def test_generate_explicit_columns_and_custom_output() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 10)
        out = Path(tmp) / "sheets" / "sheet.png"
        result = generate_contact_sheet(d, output_path=out, columns=5)
        assert (result.columns, result.rows) == (5, 2)
        assert out.exists()


def test_generate_with_labels_and_timestamps() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 4, with_metadata=True)
        plain = generate_contact_sheet(d, output_path=d / "plain.png")
        labelled = generate_contact_sheet(
            d, output_path=d / "labelled.png", labels=True, timestamps=True
        )
        # Captions add vertical space, so the labelled sheet must be taller.
        assert labelled.height > plain.height


def test_generate_transparent_background() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 4)
        result = generate_contact_sheet(d, background="transparent")
        with Image.open(result.output_path) as sheet:
            # A corner pixel (in the padding) should be fully transparent.
            assert sheet.convert("RGBA").getpixel((0, 0))[3] == 0


def test_generate_webp_format() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 4)
        result = generate_contact_sheet(d, image_format="webp")
        assert result.output_path.suffix == ".webp"
        assert result.output_path.exists()


def test_generate_progress_callback() -> None:
    if _skip_if_no_pil():
        return
    seen: list[tuple[int, int]] = []
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_frames(d, 6)
        generate_contact_sheet(d, progress_callback=lambda c, t: seen.append((c, t)))
    assert seen[0] == (1, 6)
    assert seen[-1] == (6, 6)


def test_generate_empty_directory_raises() -> None:
    if _skip_if_no_pil():
        return
    with tempfile.TemporaryDirectory() as tmp:
        raised = False
        try:
            generate_contact_sheet(tmp)
        except NoFramesFoundError:
            raised = True
        assert raised


if __name__ == "__main__":
    from factories import run_module_tests

    raise SystemExit(run_module_tests(dict(globals())))
