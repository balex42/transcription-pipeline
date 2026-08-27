"""Internal absolute-timestamp reconciliation for segmented ASR backends."""

from __future__ import annotations

from meeting_transcriber.models import ASRWord, AudioSegment


def reconcile_segment_words(
    segments: list[AudioSegment], words_by_segment: dict[int, list[ASRWord]]
) -> list[ASRWord]:
    """Offset segment-local words and retain one owner for each overlap word."""
    merged: list[ASRWord] = []
    for index, segment in enumerate(segments):
        previous_boundary = (
            float("-inf")
            if index == 0
            else (segments[index - 1].end + segment.start) / 2
        )
        next_boundary = (
            float("inf")
            if index == len(segments) - 1
            else (segment.end + segments[index + 1].start) / 2
        )
        for word in words_by_segment.get(segment.index, []):
            start = word.start if word.start is not None else 0.0
            absolute_start = min(max(segment.start + start, segment.start), segment.end)
            absolute_end = min(max(segment.start + word.end, absolute_start), segment.end)
            if absolute_end < previous_boundary or absolute_end >= next_boundary:
                continue
            merged.append(
                ASRWord(
                    text=word.text,
                    start=absolute_start,
                    end=absolute_end,
                    confidence=word.confidence,
                )
            )
    return sorted(merged, key=lambda item: (item.end, item.start or item.end))
