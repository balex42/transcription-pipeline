"""Primeline's German NeMo FastConformer TDT adapter."""

from __future__ import annotations

from collections.abc import Sequence

from speech_transcriber.config import PRIMELINE_MODEL_FILE
from speech_transcriber.models import ASRWord
from speech_transcriber.transcription import nemo_support
from speech_transcriber.transcription.nemo_segmented import (
    SegmentedNeMoTranscriber,
    flatten_nemo_words,
)


class PrimelineTranscriber(SegmentedNeMoTranscriber):
    """Transcribe Primeline in bounded segments with native word timestamps."""

    backend_name = "Primeline"
    checkpoint_file = PRIMELINE_MODEL_FILE

    def _restore_model(self, checkpoint: str) -> object:
        return _restore_primeline_model(checkpoint, self.device)


def resolve_primeline_model_path(model: str) -> str:
    """Locate Primeline's checkpoint without a runtime Hub lookup."""
    checkpoint, _snapshot = nemo_support.resolve_checkpoint_path(
        model, PRIMELINE_MODEL_FILE, subject="Primeline model"
    )
    return checkpoint


def _restore_primeline_model(model_path: str, device: str) -> object:
    """Restore and place Primeline on the requested device, importing NeMo lazily."""
    return nemo_support.restore_model(model_path, device, subject="Primeline")


def flatten_primeline_words(
    outputs: Sequence[object], *, duration_seconds: float | None = None
) -> list[ASRWord]:
    """Map one Primeline segment's native timestamps to canonical words."""
    return flatten_nemo_words(
        outputs, subject="Primeline", scope="segment", duration_seconds=duration_seconds
    )
