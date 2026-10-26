"""Typer command for GIF frame extraction.

The :func:`extract` function is shared by two entry points:

* ``extract_frames.py``      — a standalone single-command script.
* ``main.py extract-frames`` — a sub-command of the unified project CLI.

Keeping the command here (rather than duplicating it in both scripts) keeps the
wiring DRY while the heavy lifting lives in :mod:`app.gif.extractor`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from ..utils import console, format_duration, print_error, setup_logging
from .contact_sheet import generate_contact_sheet
from .extractor import extract_frames
from .utils import GifError

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Extract every frame of an animated GIF as high-quality images.",
)

contact_sheet_app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Combine extracted frames into a single contact-sheet image.",
)


class ImageFormat(str, Enum):
    """Supported output image formats."""

    png = "png"
    jpg = "jpg"
    webp = "webp"


class _SpeedColumn(ProgressColumn):
    """Render live extraction throughput in frames per second."""

    def render(self, task) -> Text:  # noqa: ANN001 - rich passes its Task object
        speed = task.speed or 0.0
        return Text(f"{speed:5.1f} fps", style="progress.data.speed")


def _print_report(result, gif_name: str) -> None:
    """Print a summary panel after a successful extraction."""
    meta = result.metadata
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Frames extracted", str(meta.frame_count))
    table.add_row("Dimensions", f"{meta.width} x {meta.height}")
    table.add_row("GIF version", meta.gif_version)
    table.add_row("Duration", f"{meta.duration_ms} ms")
    table.add_row("Average FPS", str(meta.fps))
    table.add_row("Loop count", "infinite" if meta.loop_count == 0 else str(meta.loop_count))
    table.add_row("Transparency", "yes" if meta.has_transparency else "no")
    table.add_row("Elapsed", format_duration(result.elapsed_seconds))
    table.add_row("Speed", f"{result.speed_fps} fps")
    table.add_row("Output", str(result.output_dir))
    if result.metadata_path:
        table.add_row("Metadata", str(result.metadata_path))
    console.print(Panel(table, title=f"[green]✓ {gif_name}", border_style="green"))


@app.command()
def extract(
    gif: Path = typer.Argument(
        ..., help="Path to the animated GIF to extract.", show_default=False
    ),
    output: Path = typer.Option(
        Path("./output"), "--output", "-o", help="Base output directory."
    ),
    prefix: str = typer.Option(
        "frame_", "--prefix", help="Frame filename prefix (e.g. 'frame_')."
    ),
    image_format: ImageFormat = typer.Option(
        ImageFormat.png, "--format", help="Output image format."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing frames in the output folder."
    ),
    metadata: bool = typer.Option(
        True, "--metadata/--no-metadata", help="Write a metadata.json file."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Extract every frame of an animated GIF into individual image files."""
    setup_logging(verbose)

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        _SpeedColumn(),
        TimeElapsedColumn(),
    )

    try:
        with Progress(*progress_columns, console=console, transient=True) as progress:
            task_id = progress.add_task(f"[cyan]Extracting {gif.name}", total=None)

            def on_progress(current: int, total: int) -> None:
                if progress.tasks[task_id].total is None:
                    progress.update(task_id, total=total)
                progress.update(task_id, completed=current)

            result = extract_frames(
                gif,
                output_dir=output,
                prefix=prefix,
                image_format=image_format.value,
                overwrite=overwrite,
                write_metadata_file=metadata,
                progress_callback=on_progress,
            )
    except GifError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(code=130)

    _print_report(result, gif.name)


def run() -> None:
    """Console entry point for the standalone ``extract_frames.py`` script."""
    app()


# ---------------------------------------------------------------------------
# Contact sheet command (shared by main.py and the contact_sheet.py script)
# ---------------------------------------------------------------------------
def _print_contact_sheet_report(result, source: str) -> None:
    """Print a summary panel after a contact sheet is generated."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Total frames", str(result.frame_count))
    table.add_row("Grid", f"{result.columns} x {result.rows} (cols x rows)")
    table.add_row("Thumbnail", f"{result.thumbnail_size[0]} x {result.thumbnail_size[1]} px")
    table.add_row("Output resolution", f"{result.resolution} px")
    table.add_row("Processing time", format_duration(result.elapsed_seconds))
    table.add_row("Output", str(result.output_path))
    console.print(Panel(table, title=f"[green]✓ contact sheet · {source}", border_style="green"))


@contact_sheet_app.command()
def contact_sheet(
    frames_dir: Path = typer.Argument(
        ...,
        help="Directory of extracted frames (e.g. output/animation).",
        show_default=False,
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output image path (default: <dir>/contact_sheet.png)."
    ),
    columns: int = typer.Option(
        None, "--columns", min=1, help="Number of columns. Auto (near-square) if omitted."
    ),
    padding: int = typer.Option(
        8, "--padding", min=0, help="Gap in pixels between cells and around edges."
    ),
    thumbnail_width: int = typer.Option(
        180, "--thumbnail-width", min=1, help="Per-frame thumbnail width in pixels."
    ),
    labels: bool = typer.Option(
        False, "--labels", help="Draw 'Frame N' captions under each frame."
    ),
    timestamps: bool = typer.Option(
        False, "--timestamps", help="Draw MM:SS.cc captions (needs metadata.json)."
    ),
    background: str = typer.Option(
        "white", "--background", help="Background: 'white', 'transparent' or a colour/#hex."
    ),
    image_format: ImageFormat = typer.Option(
        ImageFormat.png, "--format", help="Output image format."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Combine a folder of extracted frames into a single contact-sheet image."""
    setup_logging(verbose)

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        _SpeedColumn(),
        TimeElapsedColumn(),
    )

    try:
        with Progress(*progress_columns, console=console, transient=True) as progress:
            task_id = progress.add_task(f"[cyan]Building sheet · {frames_dir.name}", total=None)

            def on_progress(current: int, total: int) -> None:
                if progress.tasks[task_id].total is None:
                    progress.update(task_id, total=total)
                progress.update(task_id, completed=current)

            result = generate_contact_sheet(
                frames_dir,
                output_path=output,
                columns=columns,
                thumbnail_width=thumbnail_width,
                padding=padding,
                labels=labels,
                timestamps=timestamps,
                background=background,
                image_format=image_format.value,
                progress_callback=on_progress,
            )
    except GifError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(code=130)

    _print_contact_sheet_report(result, frames_dir.name)


def run_contact_sheet() -> None:
    """Console entry point for the standalone ``contact_sheet.py`` script."""
    contact_sheet_app()


if __name__ == "__main__":  # pragma: no cover
    run()
