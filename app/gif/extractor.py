"""Core GIF frame extraction.

The extractor streams frames **one at a time** (via Pillow's frame seeking) so
memory use stays flat even for GIFs with thousands of frames. Each frame is
composited (honouring GIF disposal methods), converted to a high-quality,
lossless image and written to disk in original dimensions with transparency
preserved.

Extensibility
-------------
Pass any number of :class:`FrameProcessor` objects to :func:`extract_frames`.
Each receives a :class:`FrameContext` (the in-memory RGBA frame plus timing and
path information) immediately after the frame is decoded. This is the seam for
future AI features — deduplication, scene-change detection, OCR, captioning,
object/face detection, CLIP embeddings or vision-LLM processing — without
changing the extraction core.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence, runtime_checkable

from .metadata import FrameMeta, GifMetadata, write_metadata
from .utils import (
    CorruptedGifError,
    EmptyGifError,
    InvalidGifError,
    OutputError,
    build_frame_filename,
    logger,
    prepare_output_dir,
    read_gif_version,
    resolve_output_format,
    validate_gif_path,
)

#: Progress callback signature: ``(current_frame, total_frames)``.
ProgressCallback = Callable[[int, int], None]


@dataclass(slots=True)
class FrameContext:
    """Everything a :class:`FrameProcessor` needs to analyse a single frame."""

    index: int                 # 1-based frame index
    image: "object"            # PIL.Image.Image in RGBA mode (typed loosely to avoid hard import)
    duration_ms: int
    timestamp_ms: int
    output_path: Path


@runtime_checkable
class FrameProcessor(Protocol):
    """Plug-in hook invoked once per extracted frame.

    Implement :meth:`process` to add downstream analysis (e.g. compute a CLIP
    embedding, run OCR, detect faces). Processors observe frames and should not
    mutate the image in place if other processors run after them.
    """

    def process(self, context: FrameContext) -> None:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class ExtractionResult:
    """The outcome of an :func:`extract_frames` run."""

    metadata: GifMetadata
    output_dir: Path
    frame_paths: list[Path] = field(default_factory=list)
    metadata_path: Path | None = None
    elapsed_seconds: float = 0.0

    @property
    def speed_fps(self) -> float:
        """Extraction throughput in frames processed per wall-clock second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return round(self.metadata.frame_count / self.elapsed_seconds, 2)


def _open_gif(gif_path: Path):
    """Open a GIF with Pillow, validating that it is a non-empty animation.

    Returns:
        A tuple of ``(image, frame_count)``.

    Raises:
        InvalidGifError: If the file is not actually a GIF.
        EmptyGifError: If the GIF has no frames.
        CorruptedGifError: If Pillow cannot decode the file.
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OutputError(
            "Pillow is required for GIF extraction. Install it with:\n"
            "  pip install -r requirements.txt"
        ) from exc

    try:
        image = Image.open(gif_path)
    except UnidentifiedImageError as exc:
        raise InvalidGifError(f"'{gif_path.name}' is not a readable image.") from exc
    except OSError as exc:
        raise CorruptedGifError(f"Could not open '{gif_path.name}': {exc}") from exc

    if (image.format or "").upper() != "GIF":
        raise InvalidGifError(
            f"'{gif_path.name}' is not a GIF (detected format: {image.format})."
        )

    frame_count = int(getattr(image, "n_frames", 1))
    if frame_count < 1:
        raise EmptyGifError(f"'{gif_path.name}' contains no frames.")
    return image, frame_count


def _save_frame(image, path: Path, format_key: str) -> None:
    """Save a single RGBA frame losslessly in the requested format.

    Args:
        image: A PIL RGBA image.
        path: Destination file path.
        format_key: Normalised format key (``png`` / ``jpg`` / ``webp``).

    Raises:
        OutputError: If writing fails (e.g. permissions).
    """
    from PIL import Image  # local import; Pillow already validated by caller

    try:
        if format_key in ("jpg", "jpeg"):
            # JPEG has no alpha channel — flatten onto white to avoid black halos.
            background = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.split()[-1] if image.mode == "RGBA" else None
            background.paste(image, mask=alpha)
            background.save(path, format="JPEG", quality=95, subsampling=0)
        elif format_key == "webp":
            # Lossless WebP keeps full quality and the alpha channel.
            image.save(path, format="WEBP", lossless=True, quality=100, method=6)
        else:  # png — lossless, alpha preserved, no quality loss
            image.save(path, format="PNG", optimize=False)
    except OSError as exc:
        raise OutputError(f"Could not write frame '{path}': {exc}") from exc


def extract_frames(
    gif_path: str | Path,
    output_dir: str | Path = "output",
    *,
    prefix: str = "frame_",
    image_format: str = "png",
    overwrite: bool = False,
    write_metadata_file: bool = True,
    processors: Sequence[FrameProcessor] = (),
    progress_callback: ProgressCallback | None = None,
) -> ExtractionResult:
    """Extract every frame of an animated GIF as high-quality images.

    Frames are written to ``<output_dir>/<gif_stem>/`` in original order and
    dimensions, with transparency preserved. A ``metadata.json`` describing the
    GIF and each frame is written alongside (unless disabled).

    Args:
        gif_path: Path to the input ``.gif`` (GIF87a or GIF89a).
        output_dir: Base output directory; a per-GIF sub-folder is created in it.
        prefix: Frame filename prefix (default ``"frame_"``).
        image_format: Output format — ``"png"`` (default), ``"jpg"`` or ``"webp"``.
        overwrite: Replace existing frames/metadata if the folder is non-empty.
        write_metadata_file: Whether to write ``metadata.json``.
        processors: Optional :class:`FrameProcessor` hooks for downstream analysis.
        progress_callback: Called as ``(current, total)`` after each frame.

    Returns:
        An :class:`ExtractionResult` with metadata, output paths and timing.

    Raises:
        GifNotFoundError, InvalidGifError, EmptyGifError, CorruptedGifError,
        UnsupportedOutputFormatError, OutputError: see :mod:`app.gif.utils`.
    """
    start = time.perf_counter()
    gif_file = validate_gif_path(Path(gif_path))
    gif_version = read_gif_version(gif_file)
    format_key, extension = resolve_output_format(image_format)

    image, frame_count = _open_gif(gif_file)
    target_dir = prepare_output_dir(
        Path(output_dir), gif_file.stem, extension, overwrite=overwrite
    )

    width, height = image.size
    loop_count = int(image.info.get("loop", 0) or 0)
    has_transparency = "transparency" in image.info or image.mode in ("RGBA", "LA", "PA")
    source_mode = image.mode

    logger.info(
        "Extracting %d frame(s) from '%s' (%dx%d, %s) -> %s",
        frame_count, gif_file.name, width, height, gif_version, target_dir,
    )

    frame_metas: list[FrameMeta] = []
    frame_paths: list[Path] = []
    timestamp_ms = 0

    try:
        for index in range(frame_count):
            image.seek(index)  # sequential seek lets Pillow composite disposal correctly
            # Convert to RGBA to preserve transparency and decouple from the palette.
            frame = image.convert("RGBA")
            duration_ms = int(image.info.get("duration", 0) or 0)

            filename = build_frame_filename(prefix, index + 1, extension)
            frame_path = target_dir / filename
            _save_frame(frame, frame_path, format_key)

            if processors:
                context = FrameContext(
                    index=index + 1,
                    image=frame,
                    duration_ms=duration_ms,
                    timestamp_ms=timestamp_ms,
                    output_path=frame_path,
                )
                for processor in processors:
                    processor.process(context)

            frame_metas.append(
                FrameMeta(
                    index=index + 1,
                    filename=filename,
                    duration_ms=duration_ms,
                    timestamp_ms=timestamp_ms,
                )
            )
            frame_paths.append(frame_path)
            timestamp_ms += duration_ms

            if progress_callback:
                progress_callback(index + 1, frame_count)
    except EOFError as exc:
        # n_frames over-reported, or the GIF is truncated mid-stream.
        raise CorruptedGifError(
            f"'{gif_file.name}' ended unexpectedly after {len(frame_metas)} frame(s); "
            "the file may be truncated or corrupted."
        ) from exc
    except OSError as exc:
        raise CorruptedGifError(
            f"Failed to decode '{gif_file.name}': {exc}"
        ) from exc
    finally:
        image.close()

    if not frame_metas:
        raise EmptyGifError(f"No frames could be extracted from '{gif_file.name}'.")

    total_duration_ms = timestamp_ms
    metadata = GifMetadata(
        filename=gif_file.name,
        width=width,
        height=height,
        frame_count=len(frame_metas),
        duration_ms=total_duration_ms,
        fps=GifMetadata.compute_fps(len(frame_metas), total_duration_ms),
        loop_count=loop_count,
        gif_version=gif_version,
        mode=source_mode,
        has_transparency=has_transparency,
        frames=frame_metas,
    )

    metadata_path: Path | None = None
    if write_metadata_file:
        metadata_path = write_metadata(metadata, target_dir / "metadata.json")

    elapsed = time.perf_counter() - start
    return ExtractionResult(
        metadata=metadata,
        output_dir=target_dir,
        frame_paths=frame_paths,
        metadata_path=metadata_path,
        elapsed_seconds=elapsed,
    )
