"""Writers that turn a :class:`~app.transcription.TranscriptionResult` into files.

Each writer takes the result plus a destination path and returns the path it
wrote, so the CLI can simply collect the returned paths for reporting.

Supported formats: ``.txt``, ``.srt``, ``.vtt``, ``.json``, ``.md``, ``.csv``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .transcription import TranscriptionResult
from .utils import format_timestamp, get_logger

logger = get_logger()


def write_txt(result: TranscriptionResult, path: Path) -> Path:
    """Write the plain transcript, one segment per line."""
    lines = [seg.text.strip() for seg in result.segments if seg.text.strip()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_srt(result: TranscriptionResult, path: Path) -> Path:
    """Write a SubRip (``.srt``) subtitle file."""
    blocks: list[str] = []
    for index, seg in enumerate(result.segments, start=1):
        start = format_timestamp(seg.start, decimal_marker=",")
        end = format_timestamp(seg.end, decimal_marker=",")
        blocks.append(f"{index}\n{start} --> {end}\n{seg.text.strip()}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def write_vtt(result: TranscriptionResult, path: Path) -> Path:
    """Write a WebVTT (``.vtt``) subtitle file."""
    blocks: list[str] = ["WEBVTT\n"]
    for seg in result.segments:
        start = format_timestamp(seg.start, decimal_marker=".")
        end = format_timestamp(seg.end, decimal_marker=".")
        blocks.append(f"{start} --> {end}\n{seg.text.strip()}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def write_json(result: TranscriptionResult, path: Path) -> Path:
    """Write the structured ``.json`` output (segments + metadata)."""
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_markdown(result: TranscriptionResult, path: Path) -> Path:
    """Write human-friendly Markdown notes with a metadata header."""
    meta = result.to_dict()["metadata"]
    source_name = Path(result.source_file).name if result.source_file else "transcript"

    lines: list[str] = [
        f"# Transcript — {source_name}",
        "",
        f"- **Language:** {meta['language']} "
        f"(p={meta['language_probability']})",
        f"- **Model:** {meta['model']} ({meta['device']}, {meta['compute_type']})",
        f"- **Duration:** {meta['audio_duration']}s",
        f"- **Processing time:** {meta['elapsed_seconds']}s "
        f"({meta['speed_realtime_factor']}× realtime)",
        "",
    ]
    if result.summary:
        lines += ["## Summary", "", result.summary.strip(), ""]

    lines += ["## Transcript", ""]
    for seg in result.segments:
        start = format_timestamp(seg.start, decimal_marker=".")
        lines.append(f"**[{start}]** {seg.text.strip()}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_csv(result: TranscriptionResult, path: Path) -> Path:
    """Write a ``.csv`` with one row per segment and timestamp columns."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "start", "end", "duration", "confidence", "text"])
        for index, seg in enumerate(result.segments, start=1):
            writer.writerow(
                [
                    index,
                    round(seg.start, 3),
                    round(seg.end, 3),
                    round(seg.end - seg.start, 3),
                    seg.confidence if seg.confidence is not None else "",
                    seg.text.strip(),
                ]
            )
    return path


#: Maps a format key to its writer and canonical file extension.
WRITERS: dict[str, tuple] = {
    "txt": (write_txt, ".txt"),
    "srt": (write_srt, ".srt"),
    "vtt": (write_vtt, ".vtt"),
    "json": (write_json, ".json"),
    "md": (write_markdown, ".md"),
    "csv": (write_csv, ".csv"),
}

#: Formats always produced (matches the project requirements).
DEFAULT_FORMATS: tuple[str, ...] = ("txt", "srt", "vtt", "json")


def write_outputs(
    result: TranscriptionResult,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    """Write every requested format and return the list of created paths.

    Args:
        result: The transcription result to serialise.
        output_dir: Directory to write into (created if missing).
        stem: Base filename (without extension), e.g. the source video's stem.
        formats: Which format keys from :data:`WRITERS` to emit.

    Returns:
        Paths of all files written, in the order of *formats*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        if fmt not in WRITERS:
            logger.warning("Skipping unknown output format '%s'.", fmt)
            continue
        writer, extension = WRITERS[fmt]
        target = output_dir / f"{stem}{extension}"
        written.append(writer(result, target))
        logger.debug("Wrote %s", target)
    return written


def expected_output_path(output_dir: Path, stem: str, fmt: str = "json") -> Path:
    """Return the path a given format would be written to (for skip checks)."""
    _, extension = WRITERS[fmt]
    return output_dir / f"{stem}{extension}"
