"""Internal absolute-timestamp reconciliation for segmented ASR backends."""

from __future__ import annotations

import difflib
import re

from speech_transcriber.models import ASRWord, AudioSegment

# Number of words examined at each seam when hunting for overlap duplicates.
# A 15-second overlap at typical speech rates spans roughly 40-60 words, but
# duplicates only survive when the two segment clocks disagree, so a bounded
# window is enough and keeps matching deterministic.
SEAM_WINDOW = 24
# Minimum run length for a seam match. Single shared words ("und", "ja") are
# too common to deduplicate on safely; two or more adjacent identical tokens
# indicate the same stretch of audio decoded by both segments.
SEAM_MIN_MATCH = 2

_LEXICAL_NOISE = re.compile(r"\W+", re.UNICODE)


def _lexical_key(word: ASRWord) -> str:
    """Normalize word text for cross-segment comparison.

    Punctuation-free words get a position-unique sentinel so they never join
    a deduplication run: guessing their identity would risk dropping real
    words when both segments transcribe a term differently.
    """
    key = _LEXICAL_NOISE.sub("", word.text).casefold()
    return key if key else "\x00" + str(id(word))


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


def reconcile_segment_end_words(
    segments: list[AudioSegment], words_by_segment: dict[int, list[ASRWord]]
) -> tuple[list[ASRWord], dict[str, float]]:
    """Reconcile segmented end-only words without inventing word starts.

    Backends whose native timestamps carry only word end times (start is
    ``None``) must keep ``start`` unset after rebasing; substituting the
    segment start would fabricate acoustic boundaries this pipeline never
    measured. Ownership uses the rebased end timestamp against the same
    deterministic midpoint boundaries as :func:`reconcile_segment_words`.

    Midpoint ownership alone cannot deduplicate overlap words when the two
    segment clocks disagree at a seam: both segments then transcribe the same
    words into their own kept range. Adjacent segment word lists are therefore
    compared lexically over a bounded seam window, and a later segment's copy
    of a two-or-more-word run shared with the earlier segment's tail is
    dropped in favor of the earlier copy. The earlier copy is protected even
    when its rebased end lands beyond the midpoint, so a match never loses a
    word outright. Non-matching tokens from either side always survive.

    Returns the merged words and operational counters (seam-deduplicated and
    rebasing-clipped word ends) suitable for backend metrics.
    """
    a_protected: dict[int, set[int]] = {}
    b_dropped: dict[int, set[int]] = {}
    for index in range(len(segments) - 1):
        protected, dropped = _seam_duplicate_indices(
            words_by_segment.get(segments[index].index, []),
            words_by_segment.get(segments[index + 1].index, []),
        )
        if protected:
            a_protected[index] = protected
        if dropped:
            b_dropped[index + 1] = dropped

    merged: list[ASRWord] = []
    seam_dropped = 0
    clip_count = 0
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
        protected = a_protected.get(index, set())
        dropped_here = b_dropped.get(index, set())
        for position, word in enumerate(words_by_segment.get(segment.index, [])):
            rebased_end = segment.start + word.end
            absolute_end = min(max(rebased_end, segment.start), segment.end)
            if absolute_end != rebased_end:
                clip_count += 1
            if position in dropped_here:
                # The earlier segment owns this matched run, whether the copy
                # loses by midpoint or would have survived one.
                seam_dropped += 1
                continue
            keep = previous_boundary <= absolute_end < next_boundary
            if position in protected:
                # The earlier segment's copy of a matched run must survive
                # even when its timestamp lands beyond the midpoint.
                keep = True
            if not keep:
                continue
            merged.append(
                ASRWord(
                    text=word.text,
                    start=None,
                    end=absolute_end,
                    confidence=word.confidence,
                )
            )
    words = sorted(merged, key=lambda item: item.end)
    metrics = {
        "seam_deduplicated_words": float(seam_dropped),
        "reconciliation_clipped_word_ends": float(clip_count),
    }
    return words, metrics


def _seam_duplicate_indices(
    a_words: list[ASRWord], b_words: list[ASRWord]
) -> tuple[set[int], set[int]]:
    """Find overlap duplicate runs between two adjacent segment word lists.

    Returns the positions in ``a_words`` whose copies in ``b_words`` were
    matched (protected from midpoint filtering on the a side) and the
    positions in ``b_words`` dropped in favor of the earlier segment. Only
    runs of at least :data:`SEAM_MIN_MATCH` lexically identical words count;
    within the bounded seam window the longest such run wins so a repeated
    common phrase inside the overlap cannot shadow the true seam match.
    """
    a_window = a_words[-SEAM_WINDOW:]
    b_window = b_words[:SEAM_WINDOW]
    a_offset = len(a_words) - len(a_window)
    matcher = difflib.SequenceMatcher(
        None,
        [_lexical_key(word) for word in a_window],
        [_lexical_key(word) for word in b_window],
        autojunk=False,
    )
    best: tuple[int, int, int] | None = None
    for a_start, b_start, size in matcher.get_matching_blocks():
        if size >= SEAM_MIN_MATCH and (best is None or size > best[2]):
            best = (a_start, b_start, size)
    if best is None:
        return set(), set()
    a_start, b_start, size = best
    return (
        set(range(a_offset + a_start, a_offset + a_start + size)),
        set(range(b_start, b_start + size)),
    )