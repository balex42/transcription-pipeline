"""Build derived speaker turns without changing ASR text."""

from __future__ import annotations

from meeting_transcriber.models import AttributedWord, SpeakerTurn


class TurnBuilder:
    """Group adjacent speaker-attributed words by speaker and silence gap."""

    def __init__(self, gap_seconds: float = 1.0) -> None:
        if gap_seconds < 0:
            raise ValueError("gap_seconds must be non-negative")
        self.gap_seconds = gap_seconds

    def build(self, words: list[AttributedWord]) -> list[SpeakerTurn]:
        """Return chronological turns, preserving model text and whitespace only."""
        if not words:
            return []
        ordered = sorted(words, key=lambda word: (word.start, word.end))
        turns: list[SpeakerTurn] = []
        current = [ordered[0]]
        for word in ordered[1:]:
            previous = current[-1]
            if word.speaker != previous.speaker or word.start - previous.end > self.gap_seconds:
                turns.append(self._turn(current))
                current = [word]
            else:
                current.append(word)
        turns.append(self._turn(current))
        return turns

    @staticmethod
    def _turn(words: list[AttributedWord]) -> SpeakerTurn:
        return SpeakerTurn(
            speaker=words[0].speaker,
            start=words[0].start,
            end=words[-1].end,
            text=" ".join(word.text.strip() for word in words if word.text.strip()),
        )
