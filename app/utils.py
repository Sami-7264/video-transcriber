"""Cross-cutting helpers: logging, custom exceptions, constants and formatting.

This module deliberately has **no heavy third-party imports** at import time so
that things like ``--help`` and error messages stay instant. ``rich`` is the
only non-stdlib dependency used here and it is lightweight.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Shared console. A single instance keeps colour/width handling consistent and
# lets us route both human output and log records through the same terminal.
# ---------------------------------------------------------------------------
console: Final[Console] = Console()
_error_console: Final[Console] = Console(stderr=True)


# ---------------------------------------------------------------------------
# Supported file formats
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".ts", ".3gp"}
)
AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".alac"}
)
MEDIA_EXTENSIONS: Final[frozenset[str]] = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

#: Valid Whisper model identifiers exposed through the CLI.
SUPPORTED_MODELS: Final[tuple[str, ...]] = ("tiny", "base", "small", "medium", "large-v3")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
class TranscriberError(Exception):
    """Base class for every error raised by this application.

    Carrying a single root type lets the CLI catch *expected* failures and
    print a friendly message, while still letting genuinely unexpected bugs
    bubble up with a full traceback.
    """


class FFmpegNotFoundError(TranscriberError):
    """Raised when the FFmpeg/FFprobe binaries cannot be located on PATH."""


class InvalidInputError(TranscriberError):
    """Raised for missing paths or unsupported file formats."""


class AudioExtractionError(TranscriberError):
    """Raised when FFmpeg fails to decode/extract audio (e.g. corrupted file)."""


class ModelLoadError(TranscriberError):
    """Raised when a Whisper model cannot be downloaded or loaded."""


class TranscriptionFailedError(TranscriberError):
    """Raised when inference fails (including out-of-memory conditions)."""


class OptionalDependencyError(TranscriberError):
    """Raised when an optional feature is requested but its deps are missing."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        verbose: When ``True`` the log level is ``DEBUG`` and tracebacks are
            rendered richly; otherwise only ``INFO`` and above are shown.

    Returns:
        The configured ``"video_transcriber"`` logger.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = RichHandler(
        console=_error_console,
        rich_tracebacks=True,
        show_time=verbose,
        show_path=verbose,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    logger = logging.getLogger("video_transcriber")
    logger.setLevel(level)
    logger.handlers.clear()  # idempotent: avoid duplicate handlers on re-init
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the application logger (configuring a sane default if needed)."""
    logger = logging.getLogger("video_transcriber")
    if not logger.handlers:
        return setup_logging(verbose=False)
    return logger


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float, *, decimal_marker: str = ".") -> str:
    """Format a number of seconds as ``HH:MM:SS<marker>mmm``.

    Args:
        seconds: A non-negative time offset in seconds.
        decimal_marker: ``"."`` for WebVTT / JSON, ``","`` for SubRip (SRT).

    Returns:
        A zero-padded timestamp string, e.g. ``"00:01:02.500"``.
    """
    if seconds < 0:
        seconds = 0.0
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def format_duration(seconds: float) -> str:
    """Render an elapsed duration compactly, e.g. ``"1m 05s"`` or ``"45.2s"``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def slugify(value: str) -> str:
    """Reduce an arbitrary string to a filesystem-safe, lower-case slug."""
    keep = "-_."
    cleaned = "".join(c if (c.isalnum() or c in keep) else "-" for c in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.replace("-.", ".").replace(".-", ".")
    return cleaned.strip("-").lower() or "transcript"


def is_media_file(path: Path) -> bool:
    """Return ``True`` if *path* has a recognised audio/video extension."""
    return path.suffix.lower() in MEDIA_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Return ``True`` if *path* has a recognised video extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def print_error(message: str) -> None:
    """Print a clearly formatted, user-facing error to stderr."""
    _error_console.print(f"[bold red]Error:[/bold red] {message}")
