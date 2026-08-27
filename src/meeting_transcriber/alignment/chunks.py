"""Timestamp ownership merging for overlapping ASR chunks."""

from __future__ import annotations

from collections.abc import Mapping

from meeting_transcriber.models import ASRWord, AudioChunk


class ChunkMerger:
    """Keep each overlapped word in exactly one deterministic ownership window."""

    def __init__(self, boundary_tolerance: float = 0.005) -> None:
        self.boundary_tolerance = boundary_tolerance

    def merge(
        self, chunks: list[AudioChunk], words_by_chunk: Mapping[int, list[ASRWord]]
    ) -> list[ASRWord]:
        """Convert chunk-relative timestamps to global, deduplicated ASR words.

        A half-overlap boundary belongs to the later chunk. The final chunk owns
        its final endpoint, so words at exact recording end are retained.
        """
        result: list[ASRWord] = []
        for index, chunk in enumerate(chunks):
            lower = chunk.absolute_start
            upper = chunk.absolute_end
            if index > 0:
                previous = chunks[index - 1]
                overlap = max(0.0, previous.absolute_end - chunk.absolute_start)
                lower = chunk.absolute_start + overlap / 2.0
            if index < len(chunks) - 1:
                following = chunks[index + 1]
                overlap = max(0.0, chunk.absolute_end - following.absolute_start)
                upper = chunk.absolute_end - overlap / 2.0
            for word in words_by_chunk.get(chunk.chunk_id, []):
                absolute_end = word.end + chunk.absolute_start
                is_last = index == len(chunks) - 1
                if absolute_end + self.boundary_tolerance < lower:
                    continue
                if absolute_end > upper + self.boundary_tolerance:
                    continue
                if not is_last and absolute_end >= upper - self.boundary_tolerance:
                    continue
                absolute_start = (
                    word.start + chunk.absolute_start if word.start is not None else None
                )
                result.append(
                    ASRWord(
                        text=word.text,
                        end=absolute_end,
                        start=absolute_start,
                        chunk_id=chunk.chunk_id,
                        confidence=word.confidence,
                    )
                )
        return sorted(result, key=lambda word: (word.end, word.chunk_id or -1, word.text))
