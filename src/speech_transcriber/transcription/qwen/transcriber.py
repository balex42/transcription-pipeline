"""Qwen configuration of the reusable forced-alignment transcriber."""

from __future__ import annotations

from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.forced_alignment import (
    ForcedAligner,
    ForcedAlignmentTranscriber,
    SegmentRecognizer,
)


class QwenTranscriber(ForcedAlignmentTranscriber):
    """Apply Qwen backend defaults to the generic two-phase transcriber."""

    capabilities = TranscriberCapabilities(True, True, False, False, requires_forced_alignment=True)

    def __init__(
        self,
        recognizer: SegmentRecognizer,
        aligner: ForcedAligner,
        segment_duration: float = 240.0,
        segment_overlap: float = 15.0,
    ) -> None:
        super().__init__(
            recognizer,
            aligner,
            self.capabilities,
            segment_duration,
            segment_overlap,
        )
