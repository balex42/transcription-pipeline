"""Approximate end-only timestamps from Voxtral Realtime emission markers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from speech_transcriber.errors import VoxtralTimestampError
from speech_transcriber.models import ASRWord

STREAMING_WORD = "[STREAMING_WORD]"


def parse_voxtral_words(
    token_ids: Sequence[int],
    token_pieces: Sequence[str],
    decode: Callable[[list[int]], object],
    delay_tokens: int,
    seconds_per_token: float,
    duration_seconds: float,
    metrics: dict[str, float] | None = None,
) -> list[ASRWord]:
    """Timestamp groups closed by native markers, with a flagged EOF-tail fallback."""
    if len(token_ids) != len(token_pieces):
        raise VoxtralTimestampError("Voxtral raw token IDs and pieces have different lengths")
    if delay_tokens < 0 or seconds_per_token <= 0 or duration_seconds < 0:
        raise VoxtralTimestampError("Voxtral processor exposed invalid timestamp configuration")

    words: list[ASRWord] = []
    group: list[int] = []
    last_native_end: float | None = None

    def decode_words() -> list[str]:
        text = decode(group)
        if not isinstance(text, str):
            raise VoxtralTimestampError(
                "Voxtral processor did not decode an emission group to text"
            )
        return text.split()

    def record_group(group_words: list[str], inferred: bool) -> None:
        if metrics is None or not group_words:
            return
        name = "inferred_final_emission_groups" if inferred else "native_emission_groups"
        metrics[name] = metrics.get(name, 0.0) + 1.0
        if len(group_words) > 1:
            metrics["multi_word_emission_groups"] = (
                metrics.get("multi_word_emission_groups", 0.0) + 1.0
            )

    def emit_native_group(end: float) -> None:
        nonlocal group
        if not group:
            return
        group_words = decode_words()
        record_group(group_words, inferred=False)
        words.extend(ASRWord(text=word, end=end) for word in group_words)
        group = []

    for index, (token_id, piece) in enumerate(zip(token_ids, token_pieces, strict=True)):
        if piece != STREAMING_WORD:
            group.append(token_id)
            continue
        marker_end = min(duration_seconds, max(0.0, (index - delay_tokens) * seconds_per_token))
        emit_native_group(marker_end)
        last_native_end = marker_end

    if group:
        group_words = decode_words()
        if group_words:
            if last_native_end is None:
                raise VoxtralTimestampError(
                    "Voxtral emitted text without a [STREAMING_WORD] timestamp marker"
                )
            span = duration_seconds - last_native_end
            for index, word in enumerate(group_words, start=1):
                end = last_native_end + span * index / len(group_words)
                words.append(ASRWord(text=word, end=end))
            record_group(group_words, inferred=True)
            if metrics is not None:
                metrics["inferred_final_words"] = float(len(group_words))
    return words
