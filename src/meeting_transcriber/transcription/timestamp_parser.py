"""Parser for Granite Speech Plus timestamp-mode output."""

from __future__ import annotations

import re

from meeting_transcriber.errors import TimestampParseError
from meeting_transcriber.models import ASRWord

TIMESTAMP_PATTERN = re.compile(r"\[T:(\d+)\]")
TIMESTAMP_LIKE_PATTERN = re.compile(r"\[T:[^\]]*\]")


def parse_timestamped_words(text: str, chunk_id: int | None = None) -> list[ASRWord]:
    """Parse Granite's word-end tags into chunk-relative words.

    Granite emits timestamps in centiseconds modulo ten seconds. Tags are
    unwrapped monotonically. Silence tokens (``_``) update the next word's
    inferred-boundary hint but are intentionally omitted from transcript words.
    """
    if not text.strip():
        return []
    tags = list(TIMESTAMP_PATTERN.finditer(text))
    if not tags:
        raise TimestampParseError("Granite output contains no valid [T:N] timestamp tags")
    for timestamp_like in TIMESTAMP_LIKE_PATTERN.finditer(text):
        if TIMESTAMP_PATTERN.fullmatch(timestamp_like.group()) is None:
            raise TimestampParseError("Granite output contains malformed timestamp tags")
    if "[T:" in TIMESTAMP_LIKE_PATTERN.sub("", text):
        raise TimestampParseError("Granite output contains malformed timestamp tags")
    trailing = text[tags[-1].end() :]
    if trailing.strip():
        raise TimestampParseError("Granite output has text without a following timestamp tag")

    words: list[ASRWord] = []
    offset = 0.0
    last_boundary = 0.0
    previous_end: float | None = None
    cursor = 0
    for tag in tags:
        fragment = text[cursor : tag.start()].strip()
        cursor = tag.end()
        centiseconds = int(tag.group(1))
        if not 0 <= centiseconds < 1000:
            raise TimestampParseError(
                "Granite timestamp tags must be in the range [T:0] through [T:999]"
            )
        current = centiseconds / 100.0 + offset
        while current < last_boundary:
            offset += 10.0
            current += 10.0
        last_boundary = current
        tokens = fragment.split()
        if not tokens:
            raise TimestampParseError(
                "Granite emitted a timestamp without a preceding word or silence token"
            )
        if tokens == ["_"]:
            previous_end = current
            continue
        if "_" in tokens or len(tokens) != 1:
            raise TimestampParseError(
                "each Granite timestamp must follow exactly one word or '_' silence token"
            )
        words.append(
            ASRWord(
                text=tokens[0],
                end=current,
                chunk_id=chunk_id,
                previous_boundary=previous_end,
            )
        )
        previous_end = current
    return words
