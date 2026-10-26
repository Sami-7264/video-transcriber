"""Contact-sheet (sprite-sheet) generation from a folder of extracted frames.

Combines every frame of an animation into a single grid image so a vision model
can reason about the whole animation from one picture. Frames are streamed one
at a time (open → resize → paste → close) so the only large object held in
memory is the destination canvas itself — folders with thousands of frames work
without loading them all at once.

Grid sizing is automatic and as close to square as possible (``columns =
ceil(sqrt(n))``), or fully manual via an explicit column count.

Design notes for future work
----------------------------
The layout is computed up-front by :func:`compute_grid` / :func:`compute_layout`
and rendering is isolated in :func:`generate_contact_sheet`, leaving clean seams
for planned enhancements: multi-page sheets for very large animations, PDF
export, JPEG/WebP variants, frame highlighting, scene-boundary markers and
similarity/scene-change-based frame selection.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .metadata import GifMetadata  # noqa: F401  (re-exported concept; used in docs/tests)
from .utils import (
    ContactSheetError,
    GifNotFoundError,
    NoFramesFoundError,
    OutputError,
    logger,
    resolve_output_format,
)

#: Progress callback signature: ``(current_frame, total_frames)``.
ProgressCallback = Callable[[int, int], None]

#: Image extensions treated as candidate frames.
FRAME_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

#: Default thumbnail width in pixels (height scales to preserve aspect ratio).
DEFAULT_THUMBNAIL_WIDTH = 180
DEFAULT_PADDING = 8
_OUTPUT_STEM = "contact_sheet"
_TEXT_COLOR = (0, 0, 0, 255)


@dataclass(slots=True)
class FrameRef:
    """A frame file plus its optional timestamp (ms) from metadata."""

    path: Path
    timestamp_ms: int | None = None


@dataclass(slots=True)
class ContactSheetResult:
    """The outcome of a :func:`generate_contact_sheet` run."""

    output_path: Path
    columns: int
    rows: int
    frame_count: int
    width: int
    height: int
    thumbnail_size: tuple[int, int]
    elapsed_seconds: float

    @property
    def resolution(self) -> str:
        """Human-readable output resolution, e.g. ``"1692 x 1692"``."""
        return f"{self.width} x {self.height}"


# ---------------------------------------------------------------------------
# Pure helpers (no Pillow required — importable & testable on any environment)
# ---------------------------------------------------------------------------
def compute_grid(frame_count: int, columns: int | None = None) -> tuple[int, int]:
    """Return ``(columns, rows)`` for *frame_count* frames.

    When *columns* is ``None`` the grid is made as close to square as possible
    using ``columns = ceil(sqrt(frame_count))``; rows are then derived. Examples:
    16→(4, 4), 25→(5, 5), 81→(9, 9), 100→(10, 10), 150→(13, 12).

    Args:
        frame_count: Number of frames (must be positive).
        columns: Optional explicit column count; rows are computed from it.

    Returns:
        A ``(columns, rows)`` tuple that fully covers all frames.

    Raises:
        ContactSheetError: If *frame_count* or *columns* is invalid.
    """
    if frame_count <= 0:
        raise ContactSheetError("Cannot build a contact sheet from zero frames.")
    if columns is not None:
        if columns < 1:
            raise ContactSheetError("--columns must be a positive integer.")
        cols = min(columns, frame_count)
    else:
        cols = math.ceil(math.sqrt(frame_count))
    rows = math.ceil(frame_count / cols)
    return cols, rows


def compute_layout(
    frame_count: int,
    cell_width: int,
    cell_height: int,
    *,
    columns: int | None = None,
    padding: int = DEFAULT_PADDING,
    label_height: int = 0,
) -> dict:
    """Compute the full pixel layout of the sheet.

    Returns a dict with ``columns``, ``rows``, ``cell_total_height``, ``width``
    and ``height`` — kept separate from rendering so it can be unit-tested and
    reused (e.g. by a future paginator).
    """
    cols, rows = compute_grid(frame_count, columns)
    cell_total_h = cell_height + label_height
    width = cols * cell_width + (cols + 1) * padding
    height = rows * cell_total_h + (rows + 1) * padding
    return {
        "columns": cols,
        "rows": rows,
        "cell_total_height": cell_total_h,
        "width": width,
        "height": height,
    }


def format_timestamp_label(milliseconds: int) -> str:
    """Format a millisecond offset as ``MM:SS.cc`` (centiseconds).

    Examples: ``0 -> "00:00.00"``, ``130 -> "00:00.13"``, ``270 -> "00:00.27"``.
    """
    total_centiseconds = round(max(0, milliseconds) / 10)
    minutes, remainder = divmod(total_centiseconds, 6000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _load_timestamps(frame_dir: Path) -> dict[str, int]:
    """Return a ``{filename: timestamp_ms}`` map from ``metadata.json`` if present."""
    metadata_file = frame_dir / "metadata.json"
    if not metadata_file.exists():
        return {}
    try:
        data = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Ignoring unreadable metadata.json: %s", exc)
        return {}
    mapping: dict[str, int] = {}
    for frame in data.get("frames", []):
        name = frame.get("filename")
        if name is not None:
            mapping[name] = int(frame.get("timestamp_ms", 0) or 0)
    return mapping


def discover_frames(frame_directory: str | Path) -> list[FrameRef]:
    """Find and order the frame images inside *frame_directory*.

    Ordering prefers the sequence recorded in ``metadata.json`` (authoritative
    frame order); otherwise frames are sorted by filename (zero-padded names
    sort correctly). The contact sheet output itself is always excluded.

    Args:
        frame_directory: Directory produced by the GIF extractor (or any folder
            of frame images).

    Returns:
        Ordered list of :class:`FrameRef`.

    Raises:
        GifNotFoundError: If the directory does not exist.
        ContactSheetError: If the path is not a directory.
    """
    directory = Path(frame_directory).expanduser().resolve()
    if not directory.exists():
        raise GifNotFoundError(f"Frame directory not found: {frame_directory}")
    if not directory.is_dir():
        raise ContactSheetError(f"Not a directory: {frame_directory}")

    files = {
        entry.name: entry
        for entry in directory.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in FRAME_IMAGE_EXTENSIONS
        and not entry.stem.startswith(_OUTPUT_STEM)
    }

    timestamps = _load_timestamps(directory)
    ordered_names = [name for name in timestamps if name in files]
    if not ordered_names:
        ordered_names = sorted(files)

    return [FrameRef(path=files[name], timestamp_ms=timestamps.get(name)) for name in ordered_names]


# ---------------------------------------------------------------------------
# Rendering helpers (Pillow imported lazily inside)
# ---------------------------------------------------------------------------
def _load_font(size: int):
    """Return a TrueType font at *size*, degrading to Pillow's default."""
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)  # Pillow >= 10.1
        except TypeError:  # pragma: no cover - very old Pillow
            return ImageFont.load_default()


def _resolve_background(background: str) -> tuple[int, int, int, int]:
    """Resolve a background spec to an RGBA tuple (``transparent`` supported)."""
    if background.lower() == "transparent":
        return (0, 0, 0, 0)
    if background.lower() == "white":
        return (255, 255, 255, 255)
    from PIL import ImageColor

    try:
        r, g, b = ImageColor.getrgb(background)[:3]
        return (r, g, b, 255)
    except ValueError as exc:
        raise ContactSheetError(
            f"Unrecognised background '{background}'. Use 'white', 'transparent' "
            "or a colour name / #hex value."
        ) from exc


def _save_sheet(canvas, output_path: Path, format_key: str) -> None:
    """Save the assembled canvas in the requested format."""
    from PIL import Image

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format_key in ("jpg", "jpeg"):
            background = Image.new("RGB", canvas.size, (255, 255, 255))
            background.paste(canvas, mask=canvas.split()[-1])
            background.save(output_path, format="JPEG", quality=95, subsampling=0)
        elif format_key == "webp":
            canvas.save(output_path, format="WEBP", lossless=True, quality=100, method=6)
        else:
            canvas.save(output_path, format="PNG", optimize=False)
    except OSError as exc:
        raise OutputError(f"Could not write contact sheet '{output_path}': {exc}") from exc


def _draw_centered(draw, text: str, cell_x: int, cell_width: int, y: int, font) -> None:
    """Draw *text* horizontally centred within a cell."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    tx = cell_x + (cell_width - text_width) // 2
    draw.text((tx, y), text, fill=_TEXT_COLOR, font=font)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_contact_sheet(
    frame_directory: str | Path,
    output_path: str | Path | None = None,
    columns: int | None = None,
    thumbnail_width: int = DEFAULT_THUMBNAIL_WIDTH,
    padding: int = DEFAULT_PADDING,
    labels: bool = False,
    timestamps: bool = False,
    *,
    background: str = "white",
    image_format: str = "png",
    progress_callback: ProgressCallback | None = None,
) -> ContactSheetResult:
    """Combine a folder of frames into a single contact-sheet image.

    Args:
        frame_directory: Folder of extracted frames (e.g. ``output/animation``).
        output_path: Destination image path. Defaults to
            ``<frame_directory>/contact_sheet.<ext>``.
        columns: Explicit column count; if ``None`` an optimal near-square grid
            is computed automatically.
        thumbnail_width: Per-frame thumbnail width in px (height auto-scales).
        padding: Gap (px) between cells and around the grid edge.
        labels: Draw ``Frame N`` captions under each frame.
        timestamps: Draw ``MM:SS.cc`` captions (requires ``metadata.json``).
        background: ``"white"`` (default), ``"transparent"`` or a colour/#hex.
        image_format: Output format — ``png`` (default), ``jpg`` or ``webp``.
        progress_callback: Called as ``(current, total)`` per frame placed.

    Returns:
        A :class:`ContactSheetResult` describing the generated sheet.

    Raises:
        GifNotFoundError, NoFramesFoundError, ContactSheetError, OutputError,
        UnsupportedOutputFormatError: see :mod:`app.gif.utils`.
    """
    start = time.perf_counter()
    from PIL import Image, ImageDraw

    frames = discover_frames(frame_directory)
    if not frames:
        raise NoFramesFoundError(
            f"No frame images found in '{frame_directory}'. "
            "Extract frames first with the GIF extractor."
        )
    frame_count = len(frames)

    if timestamps and not any(f.timestamp_ms is not None for f in frames):
        logger.warning(
            "--timestamps requested but no metadata.json timestamps were found; "
            "captions will show '--:--.--'."
        )

    # Cell dimensions are derived from the first frame's aspect ratio so every
    # cell is uniform; individual frames are scaled to fit without cropping.
    try:
        with Image.open(frames[0].path) as first:
            first_w, first_h = first.size
    except OSError as exc:
        raise ContactSheetError(f"Could not read frame '{frames[0].path}': {exc}") from exc

    cell_w = max(1, int(thumbnail_width))
    cell_h = max(1, round(cell_w * first_h / first_w))

    # Caption area sizing.
    font = None
    line_height = 0
    label_height = 0
    if labels or timestamps:
        font_size = max(11, cell_w // 12)
        font = _load_font(font_size)
        line_count = int(labels) + int(timestamps)
        line_height = font_size + 4
        label_height = line_count * line_height + 4

    layout = compute_layout(
        frame_count, cell_w, cell_h,
        columns=columns, padding=padding, label_height=label_height,
    )
    cols, rows = layout["columns"], layout["rows"]
    cell_total_h = layout["cell_total_height"]

    background_rgba = _resolve_background(background)
    canvas = Image.new("RGBA", (layout["width"], layout["height"]), background_rgba)
    draw = ImageDraw.Draw(canvas)

    logger.info(
        "Building %dx%d contact sheet from %d frame(s) -> %dx%d px",
        cols, rows, frame_count, layout["width"], layout["height"],
    )

    for index, frame_ref in enumerate(frames):
        col = index % cols
        row = index // cols
        cell_x = padding + col * (cell_w + padding)
        cell_y = padding + row * (cell_total_h + padding)

        try:
            with Image.open(frame_ref.path) as raw:
                frame = raw.convert("RGBA")
                fw, fh = frame.size
                scale = min(cell_w / fw, cell_h / fh)
                new_w = max(1, round(fw * scale))
                new_h = max(1, round(fh * scale))
                thumb = frame.resize((new_w, new_h), Image.LANCZOS)
        except OSError as exc:
            raise ContactSheetError(
                f"Could not read frame '{frame_ref.path}': {exc}"
            ) from exc

        offset_x = cell_x + (cell_w - new_w) // 2
        offset_y = cell_y + (cell_h - new_h) // 2
        canvas.paste(thumb, (offset_x, offset_y), thumb)

        if font is not None:
            text_y = cell_y + cell_h + 2
            if labels:
                _draw_centered(draw, f"Frame {index + 1}", cell_x, cell_w, text_y, font)
                text_y += line_height
            if timestamps:
                caption = (
                    format_timestamp_label(frame_ref.timestamp_ms)
                    if frame_ref.timestamp_ms is not None
                    else "--:--.--"
                )
                _draw_centered(draw, caption, cell_x, cell_w, text_y, font)

        if progress_callback:
            progress_callback(index + 1, frame_count)

    # Resolve output path & format.
    format_key, extension = resolve_output_format(image_format)
    if output_path is None:
        out_path = Path(frame_directory).expanduser().resolve() / f"{_OUTPUT_STEM}{extension}"
    else:
        out_path = Path(output_path).expanduser()
        if out_path.suffix:
            format_key, extension = resolve_output_format(out_path.suffix.lstrip("."))

    _save_sheet(canvas, out_path, format_key)
    elapsed = time.perf_counter() - start

    return ContactSheetResult(
        output_path=out_path,
        columns=cols,
        rows=rows,
        frame_count=frame_count,
        width=layout["width"],
        height=layout["height"],
        thumbnail_size=(cell_w, cell_h),
        elapsed_seconds=elapsed,
    )
