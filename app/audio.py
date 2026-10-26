"""Audio handling: locate FFmpeg, probe media and extract a Whisper-ready track.

We shell out to the FFmpeg/FFprobe binaries via :mod:`subprocess` rather than a
Python binding. This keeps the dependency footprint small and matches how most
users already have FFmpeg installed (a system package on PATH).

Whisper models expect 16 kHz mono PCM audio, so extraction always normalises to
that format regardless of the source.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .utils import (
    AudioExtractionError,
    FFmpegNotFoundError,
    InvalidInputError,
    MEDIA_EXTENSIONS,
    get_logger,
)

logger = get_logger()

#: Whisper's required input format.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


@dataclass(frozen=True, slots=True)
class SilenceSpan:
    """A detected span of silence, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def find_ffmpeg() -> str:
    """Return the path to the ``ffmpeg`` binary or raise.

    Raises:
        FFmpegNotFoundError: If ``ffmpeg`` is not discoverable on ``PATH``.
    """
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError(
            "FFmpeg was not found on your system PATH.\n"
            "  • macOS:   brew install ffmpeg\n"
            "  • Ubuntu:  sudo apt install ffmpeg\n"
            "  • Windows: winget install Gyan.FFmpeg\n"
            "See the README's 'FFmpeg installation' section for details."
        )
    return path


def find_ffprobe() -> str:
    """Return the path to the ``ffprobe`` binary or raise.

    Raises:
        FFmpegNotFoundError: If ``ffprobe`` is not discoverable on ``PATH``.
    """
    path = shutil.which("ffprobe")
    if not path:
        raise FFmpegNotFoundError(
            "FFprobe (part of FFmpeg) was not found on your PATH. "
            "Install the full FFmpeg suite — see the README for instructions."
        )
    return path


def ensure_ffmpeg_available() -> None:
    """Validate that both ``ffmpeg`` and ``ffprobe`` are installed.

    Raises:
        FFmpegNotFoundError: If either binary is missing.
    """
    find_ffmpeg()
    find_ffprobe()


def validate_input_file(path: Path) -> Path:
    """Resolve and validate an input media file.

    Args:
        path: Candidate file path.

    Returns:
        The resolved, existing path.

    Raises:
        InvalidInputError: If the path is missing, not a file, or has an
            unrecognised extension.
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise InvalidInputError(f"File not found: {path}")
    if not resolved.is_file():
        raise InvalidInputError(f"Not a file: {path}")
    if resolved.suffix.lower() not in MEDIA_EXTENSIONS:
        raise InvalidInputError(
            f"Unsupported format '{resolved.suffix}'. Supported extensions: "
            f"{', '.join(sorted(MEDIA_EXTENSIONS))}"
        )
    return resolved


def get_media_duration(path: Path) -> float:
    """Return the duration of a media file in seconds via ``ffprobe``.

    Args:
        path: Path to a media file.

    Returns:
        Duration in seconds, or ``0.0`` if it could not be determined.

    Raises:
        FFmpegNotFoundError: If ``ffprobe`` is unavailable.
    """
    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout or "{}")
        return float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
        logger.debug("Could not probe duration for %s: %s", path, exc)
        return 0.0


def probe_media(path: Path) -> dict:
    """Return a small dict of useful media metadata using ``ffprobe``.

    Keys: ``duration`` (float seconds), ``has_audio`` (bool), ``codec`` (str|None).
    """
    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "stream=codec_type,codec_name:format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout or "{}")
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise AudioExtractionError(
            f"Failed to read media metadata from '{path.name}'. "
            "The file may be corrupted or in an unsupported container."
        ) from exc

    streams = payload.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    return {
        "duration": float(payload.get("format", {}).get("duration", 0.0) or 0.0),
        "has_audio": bool(audio_streams),
        "codec": audio_streams[0].get("codec_name") if audio_streams else None,
    }


def extract_audio(
    input_path: Path,
    output_path: Path | None = None,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Path:
    """Extract and normalise audio to 16 kHz mono WAV for Whisper.

    Args:
        input_path: Source media file (audio or video).
        output_path: Destination ``.wav`` path. If ``None`` a temporary file is
            created and its path returned (the caller owns cleanup).
        sample_rate: Output sample rate in Hz (Whisper expects 16 kHz).

    Returns:
        Path to the extracted WAV file.

    Raises:
        AudioExtractionError: If the source has no audio or FFmpeg fails.
        FFmpegNotFoundError: If ``ffmpeg`` is unavailable.
    """
    ffmpeg = find_ffmpeg()

    metadata = probe_media(input_path)
    if not metadata["has_audio"]:
        raise AudioExtractionError(
            f"'{input_path.name}' contains no audio stream — nothing to transcribe."
        )

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        output_path = Path(tmp.name)

    cmd = [
        ffmpeg,
        "-y",                       # overwrite output
        "-i", str(input_path),
        "-vn",                      # drop any video stream
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",     # uncompressed 16-bit PCM
        "-loglevel", "error",
        str(output_path),
    ]
    logger.debug("Extracting audio: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise AudioExtractionError(
            f"FFmpeg failed to extract audio from '{input_path.name}'. "
            "The file may be corrupted or use an unsupported codec.\n"
            f"FFmpeg said: {stderr or '(no error output)'}"
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioExtractionError(
            f"Audio extraction produced an empty file for '{input_path.name}'."
        )
    return output_path


def detect_silence(
    path: Path,
    *,
    noise_threshold_db: float = -30.0,
    min_silence_duration: float = 1.0,
) -> list[SilenceSpan]:
    """Detect silent spans using FFmpeg's ``silencedetect`` filter.

    Args:
        path: Media file to analyse.
        noise_threshold_db: Levels below this (in dBFS) count as silence.
        min_silence_duration: Minimum span length, in seconds, to report.

    Returns:
        A list of :class:`SilenceSpan` objects ordered by start time.
    """
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-i", str(path),
        "-af", f"silencedetect=noise={noise_threshold_db}dB:d={min_silence_duration}",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.debug("silencedetect failed for %s: %s", path, exc)
        return []

    # FFmpeg writes silencedetect markers to stderr.
    text = result.stderr or ""
    starts = [float(m) for m in re.findall(r"silence_start:\s*([0-9.]+)", text)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([0-9.]+)", text)]
    spans = [SilenceSpan(start=s, end=e) for s, e in zip(starts, ends)]
    return spans
