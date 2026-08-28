"""Human-readable plain-text transcript export."""

from __future__ import annotations

from pathlib import Path

from speech_transcriber.models import Transcript


class TextTranscriptExporter:
    """Export speaker turns as plain text without altering ASR text."""

    def export(self, transcript: Transcript, destination: Path) -> None:
        """Write one speaker turn per line."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        blocks = [
            f"[{_format_time(turn.start)} - {_format_time(turn.end)}] {turn.speaker}\n{turn.text}"
            for turn in transcript.turns
        ]
        destination.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def _format_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{remainder:06.3f}"
