"""Faster Whisper integration: device resolution, model loading and inference.

The heavy ``faster_whisper`` import is performed lazily inside the functions
that need it so that ``--help`` and validation errors remain fast and do not
require the (large) inference stack to be importable.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .utils import (
    ModelLoadError,
    SUPPORTED_MODELS,
    TranscriptionFailedError,
    get_logger,
)

logger = get_logger()

#: Signature for the optional progress callback: ``(processed_s, total_s)``.
ProgressCallback = Callable[[float, float], None]

DeviceChoice = Literal["cpu", "cuda", "auto"]


# ---------------------------------------------------------------------------
# Data model — plain dataclasses so results are easy to serialise and test.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Word:
    """A single word with timing and model probability."""

    start: float
    end: float
    word: str
    probability: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Segment:
    """A contiguous span of transcribed speech."""

    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None  # populated by optional diarization

    @property
    def confidence(self) -> float | None:
        """A 0–1 confidence estimate derived from ``avg_logprob``.

        Whisper exposes the average token log-probability; exponentiating it
        gives an intuitive (if approximate) probability-like score.
        """
        if self.avg_logprob is None:
            return None
        return round(min(1.0, max(0.0, math.exp(self.avg_logprob))), 4)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
            "compression_ratio": self.compression_ratio,
            "words": [w.to_dict() for w in self.words] if self.words else [],
        }


@dataclass(slots=True)
class TranscriptionResult:
    """The full outcome of transcribing one media file."""

    segments: list[Segment]
    language: str
    language_probability: float
    audio_duration: float
    model: str
    device: str
    compute_type: str
    source_file: str = ""
    elapsed_seconds: float = 0.0
    # Optional enrichments populated by nice-to-have features.
    summary: str | None = None

    @property
    def text(self) -> str:
        """The complete transcript as a single normalised string."""
        return " ".join(seg.text.strip() for seg in self.segments).strip()

    @property
    def speed(self) -> float:
        """Realtime factor (audio seconds processed per wall-clock second)."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.audio_duration / self.elapsed_seconds

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dictionary (matches transcript.json)."""
        return {
            "metadata": {
                "source_file": self.source_file,
                "language": self.language,
                "language_probability": round(self.language_probability, 4),
                "model": self.model,
                "device": self.device,
                "compute_type": self.compute_type,
                "audio_duration": round(self.audio_duration, 3),
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "speed_realtime_factor": round(self.speed, 2),
            },
            "summary": self.summary,
            "segments": [seg.to_dict() for seg in self.segments],
        }


# ---------------------------------------------------------------------------
# Device / compute-type resolution
# ---------------------------------------------------------------------------
def cuda_is_available() -> bool:
    """Return ``True`` if CTranslate2 reports a usable CUDA device.

    This relies on CTranslate2 (Faster Whisper's backend) rather than PyTorch,
    since PyTorch is not a dependency of this project.
    """
    try:
        import ctranslate2  # local import; only needed for detection

        return ctranslate2.get_cuda_device_count() > 0
    except Exception as exc:  # pragma: no cover - depends on local hardware
        logger.debug("CUDA detection failed, assuming CPU: %s", exc)
        return False


def resolve_device(
    device: DeviceChoice = "auto",
    compute_type: str | None = None,
) -> tuple[str, str]:
    """Resolve the effective device and compute type.

    Args:
        device: ``"cpu"``, ``"cuda"`` or ``"auto"`` (detect GPU, fall back to CPU).
        compute_type: Explicit compute type (e.g. ``"float16"``, ``"int8"``).
            When ``None`` a sensible default is chosen per device.

    Returns:
        A ``(device, compute_type)`` tuple ready to pass to ``WhisperModel``.
    """
    if device == "auto":
        resolved_device = "cuda" if cuda_is_available() else "cpu"
        if resolved_device == "cuda":
            logger.debug("Auto-detected CUDA GPU.")
        else:
            logger.debug("No CUDA GPU detected; using CPU.")
    elif device == "cuda" and not cuda_is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        resolved_device = "cpu"
    else:
        resolved_device = device

    if compute_type:
        resolved_compute = compute_type
    elif resolved_device == "cuda":
        resolved_compute = "float16"
    else:
        # int8 is dramatically faster than float32 on CPU with minimal quality loss.
        resolved_compute = "int8"

    return resolved_device, resolved_compute


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------
class Transcriber:
    """A thin, well-behaved wrapper around ``faster_whisper.WhisperModel``."""

    def __init__(
        self,
        model_name: str = "base",
        *,
        device: DeviceChoice = "auto",
        compute_type: str | None = None,
        download_root: str | None = None,
    ) -> None:
        """Create a transcriber and load the model.

        Args:
            model_name: One of :data:`app.utils.SUPPORTED_MODELS`.
            device: Target device or ``"auto"``.
            compute_type: Override CTranslate2 compute type.
            download_root: Custom cache directory for downloaded models.

        Raises:
            ModelLoadError: If the model id is invalid or loading/downloading fails.
        """
        if model_name not in SUPPORTED_MODELS:
            raise ModelLoadError(
                f"Unsupported model '{model_name}'. "
                f"Choose one of: {', '.join(SUPPORTED_MODELS)}"
            )

        self.model_name = model_name
        self.device, self.compute_type = resolve_device(device, compute_type)
        self._model = self._load_model(download_root)

    def _load_model(self, download_root: str | None):
        """Import Faster Whisper and instantiate the model with clear errors."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelLoadError(
                "The 'faster-whisper' package is not installed.\n"
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        logger.info(
            "Loading model '%s' on %s (%s)…",
            self.model_name,
            self.device,
            self.compute_type,
        )
        try:
            return WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=download_root,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cublas" in message:
                raise ModelLoadError(
                    f"Ran out of GPU memory loading '{self.model_name}'. "
                    "Try a smaller model or '--device cpu'."
                ) from exc
            if "connection" in message or "download" in message or "not found" in message:
                raise ModelLoadError(
                    f"Could not download model '{self.model_name}'. "
                    "Check your internet connection; the first run needs to fetch "
                    "the model from Hugging Face."
                ) from exc
            raise ModelLoadError(
                f"Failed to load model '{self.model_name}': {exc}"
            ) from exc

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file into a :class:`TranscriptionResult`.

        Args:
            audio_path: Path to a (decoded) audio file.
            language: ISO language code, or ``None`` to auto-detect.
            beam_size: Beam search width (higher = more accurate, slower).
            word_timestamps: Emit per-word timings when supported.
            vad_filter: Apply voice-activity detection to skip non-speech.
            progress_callback: Called as ``(processed_s, total_s)`` while iterating
                segments so the caller can render a progress bar.

        Returns:
            A populated :class:`TranscriptionResult`.

        Raises:
            TranscriptionFailedError: On inference failure, including OOM.
        """
        start_time = time.perf_counter()
        try:
            segment_iter, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
            )
        except Exception as exc:
            raise self._wrap_inference_error(exc) from exc

        total = float(getattr(info, "duration", 0.0) or 0.0)
        segments: list[Segment] = []

        try:
            for raw in segment_iter:  # generator → inference happens here
                words = [
                    Word(
                        start=float(w.start),
                        end=float(w.end),
                        word=w.word,
                        probability=getattr(w, "probability", None),
                    )
                    for w in (raw.words or [])
                ] if word_timestamps and raw.words else []

                segments.append(
                    Segment(
                        id=raw.id,
                        start=float(raw.start),
                        end=float(raw.end),
                        text=raw.text.strip(),
                        avg_logprob=getattr(raw, "avg_logprob", None),
                        no_speech_prob=getattr(raw, "no_speech_prob", None),
                        compression_ratio=getattr(raw, "compression_ratio", None),
                        words=words,
                    )
                )
                if progress_callback and total > 0:
                    progress_callback(min(float(raw.end), total), total)
        except Exception as exc:
            raise self._wrap_inference_error(exc) from exc

        # Snap the progress bar to 100% on success.
        if progress_callback and total > 0:
            progress_callback(total, total)

        elapsed = time.perf_counter() - start_time
        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            audio_duration=total,
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            source_file=str(audio_path),
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _wrap_inference_error(exc: Exception) -> TranscriptionFailedError:
        """Translate a raw inference exception into a friendly error."""
        message = str(exc).lower()
        if "out of memory" in message or "oom" in message or "alloc" in message:
            return TranscriptionFailedError(
                "Ran out of memory during transcription. Try a smaller model "
                "(e.g. --model small), '--device cpu', or '--compute-type int8'."
            )
        return TranscriptionFailedError(f"Transcription failed: {exc}")
