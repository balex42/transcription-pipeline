"""NVIDIA NeMo Parakeet TDT adapter with native word timestamps."""

from __future__ import annotations

from collections.abc import Sequence

from speech_transcriber.config import PARAKEET_MODEL_FILE
from speech_transcriber.models import ASRWord, AudioSegment
from speech_transcriber.transcription import nemo_support
from speech_transcriber.transcription.nemo_segmented import (
    SegmentedNeMoTranscriber,
    flatten_nemo_words,
    validate_nemo_segment_words,
)


class ParakeetTranscriber(SegmentedNeMoTranscriber):
    """Transcribe Parakeet TDT segments with native word timestamps."""

    backend_name = "Parakeet"
    checkpoint_file = PARAKEET_MODEL_FILE

    def _restore_model(self, checkpoint: str) -> object:
        return _restore_parakeet_model(checkpoint, self.device)


def resolve_parakeet_model_path(model: str) -> str:
    """Locate Parakeet's checkpoint without a runtime Hub lookup."""
    checkpoint, _snapshot = nemo_support.resolve_checkpoint_path(
        model, PARAKEET_MODEL_FILE, subject="Parakeet model"
    )
    return checkpoint


def _restore_parakeet_model(model_path: str, device: str) -> object:
    """Restore and place Parakeet on the requested device, importing NeMo lazily."""
    return nemo_support.restore_model(model_path, device, subject="Parakeet")


def flatten_parakeet_words(outputs: Sequence[object]) -> list[ASRWord]:
    """Map one Parakeet segment's native timestamps to canonical words."""
    return flatten_nemo_words(outputs, subject="Parakeet", scope="segment")


def validate_parakeet_segment_words(
    words: Sequence[ASRWord], segment: AudioSegment
) -> list[ASRWord]:
    """Validate segment-local Parakeet timestamps before reconciliation."""
    return validate_nemo_segment_words(words, segment, subject="Parakeet")
