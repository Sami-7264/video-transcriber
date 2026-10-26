"""The Typer command-line interface — the user-facing entry point.

A single command keeps the documented usage (``python transcribe.py video.mp4``)
working without a sub-command name, while still exposing batch mode, optional
enhancements and a transcript-search mode through flags.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__
from .audio import (
    detect_silence,
    ensure_ffmpeg_available,
    extract_audio,
    validate_input_file,
)
from .features import assign_speakers, diarize, search_transcripts, summarize
from .subtitles import DEFAULT_FORMATS, expected_output_path, write_outputs
from .transcription import Transcriber, TranscriptionResult
from .utils import (
    MEDIA_EXTENSIONS,
    SUPPORTED_MODELS,
    TranscriberError,
    console,
    format_duration,
    print_error,
    setup_logging,
)

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Transcribe video/audio locally with Faster Whisper.",
)


# ---------------------------------------------------------------------------
# CLI enums (give pretty, validated --help choices)
# ---------------------------------------------------------------------------
class ModelName(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large_v3 = "large-v3"


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


class OutputFormat(str, Enum):
    txt = "txt"
    srt = "srt"
    vtt = "vtt"
    json = "json"
    md = "md"
    csv = "csv"


@dataclass(slots=True)
class FileOutcome:
    """Per-file result used to build the closing summary table."""

    name: str
    status: str  # "done" | "skipped" | "failed"
    detail: str = ""


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"video-transcriber {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------
def _collect_media_files(path: Path, recursive: bool) -> list[Path]:
    """Return the media files to process for a file or directory input."""
    if path.is_file():
        return [path]
    if path.is_dir():
        globber = path.rglob if recursive else path.glob
        files = [
            p for p in globber("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
        ]
        return sorted(files)
    return []


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def _process_file(
    media_path: Path,
    transcriber: Transcriber,
    *,
    output_dir: Path,
    formats: tuple[str, ...],
    language: str | None,
    beam_size: int,
    word_timestamps: bool,
    vad_filter: bool,
    skip_existing: bool,
    do_diarize: bool,
    hf_token: str | None,
    do_summarize: bool,
    summary_model: str,
    do_silence: bool,
    progress: Progress,
) -> FileOutcome:
    """Run the full extract → transcribe → enrich → write pipeline for one file."""
    media_path = validate_input_file(media_path)
    stem = media_path.stem

    if skip_existing and expected_output_path(output_dir, stem, "json").exists():
        console.print(f"[yellow]⏭  Skipping[/yellow] {media_path.name} (already transcribed)")
        return FileOutcome(media_path.name, "skipped")

    tmp_wav: Path | None = None
    task_id = progress.add_task(f"[cyan]{media_path.name}", total=None)
    try:
        progress.update(task_id, description=f"[cyan]Extracting audio · {media_path.name}")
        tmp_wav = extract_audio(media_path)

        if do_silence:
            spans = detect_silence(tmp_wav)
            total_silence = sum(s.duration for s in spans)
            console.print(
                f"[dim]🔇 {len(spans)} silent span(s), "
                f"{format_duration(total_silence)} total in {media_path.name}[/dim]"
            )

        progress.update(
            task_id, description=f"[cyan]Transcribing · {media_path.name}", total=100.0
        )

        def on_progress(done: float, total: float) -> None:
            if total > 0:
                progress.update(task_id, completed=min(100.0, done / total * 100.0))

        result: TranscriptionResult = transcriber.transcribe(
            tmp_wav,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            progress_callback=on_progress,
        )
        result.source_file = str(media_path)
        progress.update(task_id, completed=100.0)

        if do_diarize:
            progress.update(task_id, description=f"[cyan]Diarizing · {media_path.name}")
            turns = diarize(tmp_wav, hf_token=hf_token)
            assign_speakers(result, turns)

        if do_summarize:
            progress.update(task_id, description=f"[cyan]Summarizing · {media_path.name}")
            result.summary = summarize(result, model=summary_model)

        written = write_outputs(result, output_dir, stem, formats)
        progress.remove_task(task_id)

        _print_file_report(media_path, result, written)
        return FileOutcome(
            media_path.name,
            "done",
            f"{result.language} · {format_duration(result.elapsed_seconds)}",
        )
    finally:
        if tmp_wav and tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)


def _print_file_report(
    media_path: Path, result: TranscriptionResult, written: list[Path]
) -> None:
    """Print the per-file stats panel (language, model, speed, outputs)."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Detected language", f"{result.language} (p={result.language_probability:.2f})")
    table.add_row("Model", f"{result.model} · {result.device} · {result.compute_type}")
    table.add_row("Audio duration", format_duration(result.audio_duration))
    table.add_row("Elapsed time", format_duration(result.elapsed_seconds))
    table.add_row("Speed", f"{result.speed:.1f}× realtime")
    table.add_row("Segments", str(len(result.segments)))
    table.add_row("Outputs", "\n".join(str(p) for p in written))
    console.print(Panel(table, title=f"[green]✓ {media_path.name}", border_style="green"))


# ---------------------------------------------------------------------------
# Search mode
# ---------------------------------------------------------------------------
def _run_search(location: Path, query: str) -> None:
    """Search generated transcripts and print a results table."""
    hits = search_transcripts(location, query)
    if not hits:
        console.print(f"[yellow]No matches for '{query}'.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Search results for '{query}' ({len(hits)} hits)")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Time", style="magenta", no_wrap=True)
    table.add_column("Text")
    for hit in hits:
        table.add_row(hit.file.name, hit.timestamp, hit.text)
    console.print(table)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------
@app.command()
def main(
    path: Path = typer.Argument(
        ...,
        help="Path to a video/audio file, or a folder for batch mode.",
        show_default=False,
    ),
    model: ModelName = typer.Option(
        ModelName.base, "--model", "-m", help="Whisper model size."
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="Language code (e.g. 'en'). Omit to auto-detect."
    ),
    output: Path = typer.Option(
        Path("./output"), "--output", "-o", help="Directory for transcript files."
    ),
    device: Device = typer.Option(
        Device.auto, "--device", help="Compute device. 'auto' detects a GPU."
    ),
    beam_size: int = typer.Option(
        5, "--beam-size", min=1, max=20, help="Beam search width (higher = slower, better)."
    ),
    compute_type: str | None = typer.Option(
        None,
        "--compute-type",
        help="CTranslate2 compute type (e.g. float16, int8). Auto by default.",
    ),
    formats: list[OutputFormat] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output formats (repeatable). Defaults to txt, srt, vtt, json.",
    ),
    word_timestamps: bool = typer.Option(
        False, "--word-timestamps", help="Include per-word timings (if supported)."
    ),
    no_vad: bool = typer.Option(
        False, "--no-vad", help="Disable voice-activity-detection filtering."
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Recurse into sub-folders in batch mode."
    ),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip files that already have a JSON transcript."
    ),
    diarize_flag: bool = typer.Option(
        False, "--diarize", help="Label speakers (needs pyannote.audio + HF token)."
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", help="Hugging Face token for diarization."
    ),
    summarize_flag: bool = typer.Option(
        False, "--summarize", help="Generate a summary via a local Ollama LLM."
    ),
    summary_model: str = typer.Option(
        "llama3.2", "--summary-model", help="Ollama model name for summaries."
    ),
    silence: bool = typer.Option(
        False, "--silence", help="Report detected silent spans for each file."
    ),
    search: str | None = typer.Option(
        None, "--search", help="Search existing transcripts under PATH instead of transcribing."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    _version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """Transcribe a video/audio file (or a whole folder) using Faster Whisper."""
    setup_logging(verbose)

    # --- Search mode: no model or FFmpeg needed. --------------------------
    if search is not None:
        try:
            _run_search(path, search)
        except TranscriberError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1)
        return

    # --- Pre-flight checks. ----------------------------------------------
    try:
        ensure_ffmpeg_available()
    except TranscriberError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1)

    media_files = _collect_media_files(path.expanduser(), recursive)
    if not media_files:
        print_error(
            f"No media files found at '{path}'. "
            f"Supported: {', '.join(sorted(MEDIA_EXTENSIONS))}"
        )
        raise typer.Exit(code=1)

    selected_formats: tuple[str, ...] = (
        tuple(f.value for f in formats) if formats else DEFAULT_FORMATS
    )
    output_dir = output.expanduser()

    # --- Load the model once and reuse it across the batch. --------------
    try:
        transcriber = Transcriber(
            model.value, device=device.value, compute_type=compute_type
        )
    except TranscriberError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1)

    if len(media_files) > 1:
        console.print(f"[bold]Found {len(media_files)} media file(s) to transcribe.[/bold]")

    outcomes: list[FileOutcome] = []
    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*progress_columns, console=console, transient=True) as progress:
        for media_path in media_files:
            try:
                outcomes.append(
                    _process_file(
                        media_path,
                        transcriber,
                        output_dir=output_dir,
                        formats=selected_formats,
                        language=language,
                        beam_size=beam_size,
                        word_timestamps=word_timestamps,
                        vad_filter=not no_vad,
                        skip_existing=skip_existing,
                        do_diarize=diarize_flag,
                        hf_token=hf_token,
                        do_summarize=summarize_flag,
                        summary_model=summary_model,
                        do_silence=silence,
                        progress=progress,
                    )
                )
            except TranscriberError as exc:
                print_error(f"{media_path.name}: {exc}")
                outcomes.append(FileOutcome(media_path.name, "failed", str(exc)))
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user.[/yellow]")
                raise typer.Exit(code=130)

    if len(media_files) > 1:
        _print_batch_summary(outcomes)

    if any(o.status == "failed" for o in outcomes):
        raise typer.Exit(code=1)


def _print_batch_summary(outcomes: list[FileOutcome]) -> None:
    """Print a closing table summarising a batch run."""
    table = Table(title="Batch summary")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")
    styles = {"done": "green", "skipped": "yellow", "failed": "red"}
    for outcome in outcomes:
        style = styles.get(outcome.status, "white")
        table.add_row(outcome.name, f"[{style}]{outcome.status}[/{style}]", outcome.detail)
    console.print(table)


def run() -> None:
    """Console entry point used by ``transcribe.py`` and packaging."""
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
