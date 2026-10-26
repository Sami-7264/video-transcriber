"""Optional, gracefully-degrading enhancements.

None of these are required for core transcription. Each lazily imports its heavy
dependency and raises a friendly :class:`OptionalDependencyError` when missing,
so the base install stays light.

* :func:`diarize` / :func:`assign_speakers` — speaker labels via ``pyannote.audio``.
* :func:`summarize`                          — local-LLM summary via Ollama.
* :func:`search_transcripts`                 — substring/keyword search over outputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .transcription import TranscriptionResult
from .utils import OptionalDependencyError, format_timestamp, get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Speaker diarization (pyannote.audio)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A span of audio attributed to a single speaker."""

    start: float
    end: float
    speaker: str


def diarize(
    audio_path: Path,
    *,
    hf_token: str | None = None,
    num_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """Run speaker diarization and return ordered speaker turns.

    Args:
        audio_path: Path to the (decoded) audio file.
        hf_token: Hugging Face access token. Falls back to ``HF_TOKEN`` /
            ``HUGGINGFACE_TOKEN`` environment variables.
        num_speakers: Optional hint for the exact number of speakers.

    Returns:
        A list of :class:`SpeakerTurn` ordered by start time.

    Raises:
        OptionalDependencyError: If ``pyannote.audio`` is missing or no token
            is available.
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "Speaker diarization needs 'pyannote.audio'. Install it with:\n"
            "  pip install -r requirements-optional.txt"
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise OptionalDependencyError(
            "Diarization requires a Hugging Face token. Create one at "
            "https://hf.co/settings/tokens, accept the pyannote model terms, then "
            "set HF_TOKEN or pass --hf-token."
        )

    logger.info("Running speaker diarization (this can take a while)…")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        annotation = pipeline(str(audio_path), num_speakers=num_speakers)
    except Exception as exc:
        raise OptionalDependencyError(f"Diarization failed: {exc}") from exc

    turns = [
        SpeakerTurn(start=segment.start, end=segment.end, speaker=str(speaker))
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t.start)
    return turns


def assign_speakers(
    result: TranscriptionResult, turns: list[SpeakerTurn]
) -> TranscriptionResult:
    """Annotate each segment with the most-overlapping speaker (in place).

    Args:
        result: Transcription result whose segments will be labelled.
        turns: Speaker turns from :func:`diarize`.

    Returns:
        The same *result* object, with ``segment.speaker`` populated.
    """
    for seg in result.segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        seg.speaker = best_speaker
    return result


# ---------------------------------------------------------------------------
# Summarization (local LLM via Ollama)
# ---------------------------------------------------------------------------
def summarize(result: TranscriptionResult, *, model: str = "llama3.2") -> str:
    """Generate a short summary of the transcript using a local Ollama model.

    Args:
        result: Transcription result to summarise.
        model: Name of a model already pulled into Ollama (e.g. ``"llama3.2"``).

    Returns:
        The generated summary text.

    Raises:
        OptionalDependencyError: If the ``ollama`` package or daemon is missing.
    """
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "Summaries need the 'ollama' client. Install it with:\n"
            "  pip install -r requirements-optional.txt\n"
            "and run the Ollama daemon (https://ollama.com)."
        ) from exc

    transcript = result.text
    if not transcript:
        return ""

    prompt = (
        "Summarise the following transcript in a few concise bullet points, "
        "capturing the key topics, decisions and action items.\n\n"
        f"Transcript:\n{transcript}"
    )
    logger.info("Generating summary with Ollama model '%s'…", model)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        raise OptionalDependencyError(
            f"Summary generation failed (is the Ollama daemon running and is the "
            f"'{model}' model pulled?): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Transcript search
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SearchHit:
    """A single search match within a transcript JSON file."""

    file: Path
    start: float
    end: float
    text: str

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)


def search_transcripts(
    location: Path, query: str, *, case_sensitive: bool = False
) -> list[SearchHit]:
    """Search previously generated ``transcript.json`` files for *query*.

    Args:
        location: A ``.json`` transcript file or a directory to scan (recursively).
        query: Substring to look for within each segment's text.
        case_sensitive: Whether matching respects case.

    Returns:
        Matching segments as :class:`SearchHit` objects, ordered by file then time.
    """
    targets: list[Path]
    if location.is_dir():
        targets = sorted(location.rglob("*.json"))
    elif location.suffix.lower() == ".json":
        targets = [location]
    else:
        raise OptionalDependencyError(
            f"Search expects a .json transcript or a directory, got: {location}"
        )

    needle = query if case_sensitive else query.lower()
    hits: list[SearchHit] = []
    for target in targets:
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping unreadable transcript %s: %s", target, exc)
            continue
        for seg in data.get("segments", []):
            text = seg.get("text", "")
            haystack = text if case_sensitive else text.lower()
            if needle in haystack:
                hits.append(
                    SearchHit(
                        file=target,
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=text.strip(),
                    )
                )
    return hits
