"""Transcription stage interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from speech_transcriber.models import ASRWord, NormalizedAudio


@dataclass(frozen=True)
class TranscriberCapabilities:
    """Timestamp and text-formatting characteristics of an ASR backend."""

    word_start_timestamps: bool
    word_end_timestamps: bool
    punctuation: bool
    capitalization: bool
    streaming: bool = False
    requires_forced_alignment: bool = False


class Transcriber(Protocol):
    """Transcribe one complete normalized recording into global ASR words."""

    capabilities: TranscriberCapabilities
    model_reference: str
    dtype_name: str

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Return finalized timestamped words relative to the recording start."""

    def release(self) -> None:
        """Release model resources."""

    def load(self) -> None:
        """Load the model so callers can measure model initialization separately."""
