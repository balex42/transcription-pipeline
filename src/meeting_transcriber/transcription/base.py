"""Transcription stage interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from meeting_transcriber.models import ASRWord, AudioChunk


@dataclass(frozen=True)
class TranscriberCapabilities:
    """Timestamp and text-formatting characteristics of an ASR backend."""

    word_start_timestamps: bool
    word_end_timestamps: bool
    punctuation: bool
    capitalization: bool


class Transcriber(Protocol):
    """Transcribe one audio chunk into chunk-relative ASR words."""

    capabilities: TranscriberCapabilities
    model_reference: str
    dtype_name: str

    def transcribe(self, chunk: AudioChunk) -> list[ASRWord]:
        """Return raw words whose ``end`` is relative to ``chunk``."""

    def release(self) -> None:
        """Release model resources."""

    def load(self) -> None:
        """Load the model so callers can measure model initialization separately."""


@runtime_checkable
class BatchTranscriber(Protocol):
    """Optional ASR interface for backends with a meeting-level lifecycle."""

    def transcribe_chunks(self, chunks: list[AudioChunk]) -> dict[int, list[ASRWord]]:
        """Return chunk-relative words after processing all supplied chunks."""
