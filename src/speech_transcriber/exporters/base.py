"""Exporter interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from speech_transcriber.models import Transcript


class TranscriptExporter(Protocol):
    """Write a transcript representation to a destination."""

    def export(self, transcript: Transcript, destination: Path) -> None:
        """Serialize ``transcript`` into ``destination``."""
