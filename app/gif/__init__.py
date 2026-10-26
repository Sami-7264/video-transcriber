"""GIF frame extraction — a self-contained, extensible sub-package.

Extract every frame of an animated GIF (GIF87a / GIF89a) as a high-quality,
lossless image while preserving dimensions, transparency, colour and order.

Public API
----------
>>> from app.gif import extract_frames
>>> result = extract_frames("animation.gif", output_dir="output")
>>> result.metadata.frame_count
150

The module is organised for extension (frame deduplication, OCR, captioning,
object/face detection, CLIP embeddings, vision-LLM processing) via the
:class:`~app.gif.extractor.FrameProcessor` protocol — see ``extractor.py``.
"""

from __future__ import annotations

from .contact_sheet import (
    ContactSheetResult,
    compute_grid,
    generate_contact_sheet,
)
from .extractor import ExtractionResult, FrameContext, FrameProcessor, extract_frames
from .metadata import FrameMeta, GifMetadata

__all__ = [
    "extract_frames",
    "ExtractionResult",
    "FrameContext",
    "FrameProcessor",
    "GifMetadata",
    "FrameMeta",
    "generate_contact_sheet",
    "ContactSheetResult",
    "compute_grid",
]
