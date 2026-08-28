"""Diarization stage interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from speech_transcriber.models import DiarizationSegment


class Diarizer(Protocol):
    """Produce exclusive speaker segments over a complete recording."""

    def diarize(self, normalized_wav: Path) -> list[DiarizationSegment]:
        """Diarize the full normalized recording."""

    def release(self) -> None:
        """Release model resources."""
