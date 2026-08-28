"""Cohere configuration of the reusable forced-alignment transcriber."""

from __future__ import annotations

from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.forced_alignment import (
    ForcedAligner,
    ForcedAlignmentTranscriber,
    SegmentRecognizer,
)


class CohereTranscriber(ForcedAlignmentTranscriber):
    """Apply Cohere defaults to the generic two-phase transcriber."""

    capabilities = TranscriberCapabilities(True, True, True, True, requires_forced_alignment=True)

    def __init__(
        self,
        recognizer: SegmentRecognizer,
        aligner: ForcedAligner,
        segment_duration: float = 30.0,
        segment_overlap: float = 5.0,
        language: str = "de-DE",
        punctuation: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        super().__init__(
            recognizer,
            aligner,
            self.capabilities,
            segment_duration,
            segment_overlap,
        )
        self.backend_configuration.update(
            {
                "language": language.split("-", 1)[0].lower(),
                "punctuation": punctuation,
                "max_new_tokens": max_new_tokens,
            }
        )
