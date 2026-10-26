"""Helpers for the GIF extraction module: errors, validation and file I/O.

This module is intentionally free of any heavy imports (no Pillow at import
time) so validation and ``--help`` stay fast. It reuses the wider project's
console/formatting helpers from :mod:`app.utils` for a consistent UX while
keeping its own focused exception hierarchy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from ..utils import slugify  # reuse the project's filesystem-safe slug helper

logger = logging.getLogger("video_transcriber.gif")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GIF_EXTENSION: Final[str] = ".gif"
GIF_SIGNATURES: Final[frozenset[bytes]] = frozenset({b"GIF87a", b"GIF89a"})

#: Output image formats and their canonical file extensions.
OUTPUT_FORMATS: Final[dict[str, str]] = {
    "png": ".png",
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "webp": ".webp",
}

#: Zero-padding width for frame indices (handles up to 999,999 frames).
FRAME_INDEX_PADDING: Final[int] = 6


# ---------------------------------------------------------------------------
# Exception hierarchy (self-contained for reusability of the gif package)
# ---------------------------------------------------------------------------
class GifError(Exception):
    """Base class for every error raised by the GIF extraction module."""


class GifNotFoundError(GifError):
    """Raised when the input path does not exist or is not a file."""


class InvalidGifError(GifError):
    """Raised for non-GIF files or unsupported/unknown GIF signatures."""


class EmptyGifError(GifError):
    """Raised when a GIF contains no frames."""


class CorruptedGifError(GifError):
    """Raised when a GIF cannot be decoded (truncated/corrupted data)."""


class UnsupportedOutputFormatError(GifError):
    """Raised when an unknown output image format is requested."""


class OutputError(GifError):
    """Raised for output-directory problems (permissions, existing frames…)."""


class ContactSheetError(GifError):
    """Base class for contact-sheet generation errors."""


class NoFramesFoundError(ContactSheetError):
    """Raised when a frame directory contains no usable frame images."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_gif_path(path: Path) -> Path:
    """Resolve and validate that *path* is an existing ``.gif`` file.

    Args:
        path: Candidate input path.

    Returns:
        The resolved, existing path.

    Raises:
        GifNotFoundError: If the path is missing or is not a regular file.
        InvalidGifError: If the file does not have a ``.gif`` extension.
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise GifNotFoundError(f"File not found: {path}")
    if not resolved.is_file():
        raise GifNotFoundError(f"Not a file: {path}")
    if resolved.suffix.lower() != GIF_EXTENSION:
        raise InvalidGifError(
            f"Expected a '.gif' file but got '{resolved.suffix}': {path}"
        )
    return resolved


def read_gif_version(path: Path) -> str:
    """Return the GIF version string by reading the 6-byte file header.

    Args:
        path: Path to a candidate GIF file.

    Returns:
        Either ``"GIF87a"`` or ``"GIF89a"``.

    Raises:
        InvalidGifError: If the header is not a recognised GIF signature.
        CorruptedGifError: If the file cannot be read.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(6)
    except OSError as exc:
        raise CorruptedGifError(f"Could not read '{path.name}': {exc}") from exc

    if header not in GIF_SIGNATURES:
        raise InvalidGifError(
            f"'{path.name}' is not a valid GIF (signature was {header!r}). "
            "Only GIF87a and GIF89a are supported."
        )
    return header.decode("ascii")


def resolve_output_format(image_format: str) -> tuple[str, str]:
    """Normalise an output format and return ``(format_key, extension)``.

    Args:
        image_format: One of ``png``, ``jpg``/``jpeg`` or ``webp`` (any case).

    Returns:
        A ``(normalised_key, extension)`` tuple, e.g. ``("png", ".png")``.

    Raises:
        UnsupportedOutputFormatError: If the format is not recognised.
    """
    key = image_format.lower().lstrip(".")
    if key not in OUTPUT_FORMATS:
        raise UnsupportedOutputFormatError(
            f"Unsupported output format '{image_format}'. "
            f"Choose one of: png, jpg, webp."
        )
    return key, OUTPUT_FORMATS[key]


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------
def build_frame_filename(
    prefix: str, index: int, extension: str, *, padding: int = FRAME_INDEX_PADDING
) -> str:
    """Build a zero-padded frame filename, e.g. ``frame_000001.png``.

    Args:
        prefix: Filename prefix (e.g. ``"frame_"``).
        index: 1-based frame index.
        extension: File extension including the dot (e.g. ``".png"``).
        padding: Zero-padding width for the index.

    Returns:
        The composed filename.
    """
    return f"{prefix}{index:0{padding}d}{extension}"


def prepare_output_dir(
    base_dir: Path, name: str, extension: str, *, overwrite: bool
) -> Path:
    """Create (and optionally clear) the per-GIF output directory.

    Args:
        base_dir: Base output directory (e.g. ``./output``).
        name: Sub-folder name, typically the GIF's stem.
        extension: Frame file extension, used when clearing on overwrite.
        overwrite: If ``True``, remove existing frame/metadata files first.

    Returns:
        The prepared output directory path (``base_dir / slug(name)``).

    Raises:
        OutputError: If the directory exists with frames and ``overwrite`` is
            ``False``, or if the directory cannot be created/cleared.
    """
    target = base_dir.expanduser() / slugify(name)

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(
            f"Could not create output directory '{target}': {exc}"
        ) from exc

    existing = [p for p in target.glob(f"*{extension}")]
    metadata_file = target / "metadata.json"
    if (existing or metadata_file.exists()) and not overwrite:
        raise OutputError(
            f"Output directory '{target}' already contains frames. "
            "Use --overwrite to replace them."
        )

    if overwrite:
        try:
            for stale in existing:
                stale.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
        except OSError as exc:
            raise OutputError(
                f"Could not clear existing output in '{target}': {exc}"
            ) from exc

    return target
