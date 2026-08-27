"""Canonical versioned JSON transcript export."""

from __future__ import annotations

import json
from pathlib import Path

from meeting_transcriber.models import Transcript


class JsonTranscriptExporter:
    """Export the canonical, machine-readable transcript schema."""

    def export(self, transcript: Transcript, destination: Path) -> None:
        """Write canonical JSON. Word starts are explicitly marked inferred."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": transcript.version,
            "metadata": {
                "source": transcript.metadata.source,
                "language": transcript.language,
                "duration_seconds": transcript.metadata.duration_seconds,
                "sample_rate": transcript.metadata.sample_rate,
                "asr_backend": transcript.asr_backend,
                "asr_model": transcript.asr_model,
                "diarization_model": transcript.diarization_model,
                "word_start_note": (
                    "start is inferred for alignment; end is Granite-generated word-end timing."
                ),
            },
            "speakers": transcript.speakers,
            "turns": [
                {"speaker": turn.speaker, "start": turn.start, "end": turn.end, "text": turn.text}
                for turn in transcript.turns
            ],
            "words": [
                {
                    "text": word.text,
                    "start": word.start,
                    "end": word.end,
                    "speaker": word.speaker,
                    "start_is_inferred": word.start_is_inferred,
                }
                for word in transcript.words
            ],
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
