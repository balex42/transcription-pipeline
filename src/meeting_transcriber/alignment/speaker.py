"""Speaker assignment from ASR word intervals and exclusive diarization."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from meeting_transcriber.models import ASRWord, AttributedWord, DiarizationSegment

UNKNOWN_SPEAKER = "UNKNOWN"


class SpeakerAligner(Protocol):
    """Assign one diarization speaker to each ASR word."""

    def align(
        self, words: list[ASRWord], diarization: list[DiarizationSegment]
    ) -> list[AttributedWord]:
        """Return speaker-attributed words."""


class OverlapSpeakerAligner:
    """Assign speakers by approximate word interval overlap then timestamp fallback."""

    def __init__(self, tolerance_seconds: float = 0.25) -> None:
        self.tolerance_seconds = tolerance_seconds

    def align(
        self, words: list[ASRWord], diarization: list[DiarizationSegment]
    ) -> list[AttributedWord]:
        """Attribute words by native interval, with an end-time fallback."""
        segments = sorted(
            diarization, key=lambda segment: (segment.start, segment.end, segment.speaker)
        )
        attributed: list[AttributedWord] = []
        previous_end = 0.0
        for word in sorted(words, key=lambda item: item.end):
            start = (
                word.start
                if word.start is not None
                else previous_end
            )
            start = min(max(start, 0.0), word.end)
            speaker = self._by_overlap(start, word.end, segments) or self._at_timestamp(
                word.end, segments
            )
            attributed.append(
                AttributedWord(
                    text=word.text,
                    start=start,
                    end=word.end,
                    speaker=speaker or UNKNOWN_SPEAKER,
                    start_is_inferred=word.start is None,
                    chunk_id=word.chunk_id,
                )
            )
            previous_end = max(previous_end, word.end)
        return attributed

    @staticmethod
    def _by_overlap(start: float, end: float, segments: list[DiarizationSegment]) -> str | None:
        durations: defaultdict[str, float] = defaultdict(float)
        for segment in segments:
            overlap = min(end, segment.end) - max(start, segment.start)
            if overlap > 0:
                durations[segment.speaker] += overlap
        return max(durations, key=durations.__getitem__) if durations else None

    def _at_timestamp(self, timestamp: float, segments: list[DiarizationSegment]) -> str | None:
        candidates: list[tuple[float, str]] = []
        for segment in segments:
            if segment.start <= timestamp <= segment.end:
                return segment.speaker
            distance = min(abs(timestamp - segment.start), abs(timestamp - segment.end))
            if distance <= self.tolerance_seconds:
                candidates.append((distance, segment.speaker))
        return min(candidates)[1] if candidates else None
