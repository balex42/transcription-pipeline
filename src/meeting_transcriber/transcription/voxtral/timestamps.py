"""Approximate end-only timestamps from Voxtral Realtime emission markers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from meeting_transcriber.errors import VoxtralTimestampError
from meeting_transcriber.models import ASRWord

STREAMING_WORD = "[STREAMING_WORD]"


def parse_voxtral_words(
    token_ids: Sequence[int],
    token_pieces: Sequence[str],
    decode: Callable[[list[int]], object],
    delay_tokens: int,
    seconds_per_token: float,
    duration_seconds: float,
) -> list[ASRWord]:
    """Close emission groups at native markers using their delayed token positions."""
    if len(token_ids) != len(token_pieces):
        raise VoxtralTimestampError("Voxtral raw token IDs and pieces have different lengths")
    if delay_tokens < 0 or seconds_per_token <= 0 or duration_seconds < 0:
        raise VoxtralTimestampError("Voxtral processor exposed invalid timestamp configuration")

    words: list[ASRWord] = []
    group: list[int] = []
    marker_count = 0
    for index, (token_id, piece) in enumerate(zip(token_ids, token_pieces, strict=True)):
        if piece != STREAMING_WORD:
            group.append(token_id)
            continue
        marker_count += 1
        if not group:
            continue
        text = decode(group)
        if not isinstance(text, str):
            raise VoxtralTimestampError(
                "Voxtral processor did not decode an emission group to text"
            )
        end = min(duration_seconds, max(0.0, (index - delay_tokens) * seconds_per_token))
        words.extend(ASRWord(text=word, end=end) for word in text.split())
        group = []

    if group:
        text = decode(group)
        if not isinstance(text, str):
            raise VoxtralTimestampError(
                "Voxtral processor did not decode an emission group to text"
            )
        if text.strip():
            raise VoxtralTimestampError(
                "Voxtral emitted text without a [STREAMING_WORD] timestamp marker"
            )
    if marker_count == 0 and words:
        raise VoxtralTimestampError("Voxtral did not emit [STREAMING_WORD] timestamp markers")
    return words
