"""Internal absolute-timestamp reconciliation for segmented ASR backends."""

from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import dataclass, field

from speech_transcriber.models import ASRWord, AudioSegment

# Number of words examined at each seam when hunting for overlap duplicates.
# A 15-second overlap at typical speech rates spans roughly 40-60 words; only
# words whose rebased ends fall inside the physical overlap interval are
# considered, and this cap keeps matching deterministic.
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


@dataclass
class _SeamPlan:
    """Per-seam decisions derived from lexical overlap alignment."""

    # Positions in the earlier (a) segment to protect from midpoint filtering
    # where midpoint would otherwise lose the word on both sides.
    a_protected: set[int] = field(default_factory=set)
    # Positions in the later (b) segment to drop in favor of the a copy.
    b_dropped: set[int] = field(default_factory=set)
    # Words whose pair would have been lost entirely without protection.
    recovered: int = 0
    # Rebased end deltas (a - b) of matched pairs, for clock-offset telemetry.
    offsets: list[float] = field(default_factory=list)


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
    words into their own kept range (duplicates), or midpoint loses a word on
    both sides. Adjacent segment word lists are therefore matched lexically
    inside the physical overlap interval, and each matched run of at least
    :data:`SEAM_MIN_MATCH` words is judged by midpoint ownership: when both
    copies survive, the later segment's copy is dropped (duplicate); when
    neither survives, the earlier segment's copy is protected (recovered).
    Exactly-one-survives pairs already yield a single survivor from plain
    midpoint and stay untouched. Non-matching tokens keep midpoint ownership.

    The matcher never invents word starts and never rewrites timestamps: when
    a segment clock is off, the seam word keeps its native rebased end and
    the disagreement appears as ``seam_clock_offset_seconds`` telemetry.

    Returns the merged words and operational counters (seam matches, duplicate
    drops, recovered words, clock-offset telemetry, and rebasing-clipped word
    ends) suitable for backend metrics.
    """
    plans: dict[int, _SeamPlan] = {}
    for index in range(len(segments) - 1):
        plans[index] = _plan_seam(
            segments[index],
            segments[index + 1],
            words_by_segment.get(segments[index].index, []),
            words_by_segment.get(segments[index + 1].index, []),
        )

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
        plan_before = plans.get(index - 1)
        plan_after = plans.get(index)
        a_protected = plan_before.a_protected if plan_before is not None else set()
        b_dropped = plan_after.b_dropped if plan_after is not None else set()
        for position, word in enumerate(words_by_segment.get(segment.index, [])):
            rebased_end = segment.start + word.end
            absolute_end = min(max(rebased_end, segment.start), segment.end)
            if absolute_end != rebased_end:
                clip_count += 1
            if position in b_dropped:
                seam_dropped += 1
                continue
            keep = previous_boundary <= absolute_end < next_boundary
            if not keep and position in a_protected:
                # A run midpoint would lose on both sides keeps its earlier
                # copy so the pair still yields one survivor.
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
    offsets = [offset for plan in plans.values() for offset in plan.offsets]
    metrics = {
        "seam_match_count": float(sum(1 for plan in plans.values() if plan.offsets)),
        "seam_matched_words": float(sum(len(plan.offsets) for plan in plans.values())),
        "seam_deduplicated_words": float(seam_dropped),
        "seam_words_recovered": float(sum(plan.recovered for plan in plans.values())),
        "reconciliation_clipped_word_ends": float(clip_count),
        "seam_clock_offset_seconds": round(statistics.median(offsets), 3)
        if offsets
        else 0.0,
    }
    return words, metrics


def _plan_seam(
    segment_a: AudioSegment,
    segment_b: AudioSegment,
    a_words: list[ASRWord],
    b_words: list[ASRWord],
) -> _SeamPlan:
    """Decide seam duplicates for one adjacent segment pair.

    The physical overlap ``[b.start, a.end)`` bounds which words are seam
    candidates at all. Inside it, the rebased lexical sequences of the two
    segments are aligned; each matched run of at least
    :data:`SEAM_MIN_MATCH` words is judged by midpoint ownership, where the
    seam midpoint is ``(a.end + b.start) / 2``: a copy survives when its
    rebased end stays on its own segment's side of that midpoint.
    """
    overlap_start = segment_b.start
    overlap_end = segment_a.end
    seam_midpoint = (segment_a.end + segment_b.start) / 2

    a_candidates = _seam_candidates(
        segment_a, a_words, tail=True, overlap=(overlap_start, overlap_end)
    )
    b_candidates = _seam_candidates(
        segment_b, b_words, tail=False, overlap=(overlap_start, overlap_end)
    )

    plan = _SeamPlan()
    if not a_candidates or not b_candidates:
        return plan

    a_keys = [_lexical_key(entry[1]) for entry in a_candidates]
    b_keys = [_lexical_key(entry[1]) for entry in b_candidates]
    matcher = difflib.SequenceMatcher(None, a_keys, b_keys, autojunk=False)
    for a_start, b_start, size in matcher.get_matching_blocks():
        if size < SEAM_MIN_MATCH:
            continue
        plan.offsets.extend(
            a_candidates[a_start + i][2] - b_candidates[b_start + i][2]
            for i in range(size)
        )
        for i in range(size):
            a_position, _, a_end = a_candidates[a_start + i]
            b_position, _, b_end = b_candidates[b_start + i]
            a_survives = a_end < seam_midpoint
            b_survives = b_end >= seam_midpoint
            if a_survives and b_survives:
                plan.b_dropped.add(b_position)
            elif not a_survives and not b_survives:
                plan.a_protected.add(a_position)
                plan.recovered += 1
    return plan


def _seam_candidates(
    segment: AudioSegment,
    words: list[ASRWord],
    *,
    tail: bool,
    overlap: tuple[float, float],
) -> list[tuple[int, ASRWord, float]]:
    """Collect seam-window words whose rebased end falls inside the overlap.

    The earlier segment contributes its tail, the later segment its head;
    each entry carries the original list position, the word, and its rebased
    absolute end clipped to the owning segment.
    """
    window = words[-SEAM_WINDOW:] if tail else words[:SEAM_WINDOW]
    offset = len(words) - len(window) if tail else 0
    overlap_start, overlap_end = overlap
    candidates: list[tuple[int, ASRWord, float]] = []
    for position, word in enumerate(window):
        rebased_end = segment.start + word.end
        absolute_end = min(max(rebased_end, segment.start), segment.end)
        if overlap_start <= absolute_end < overlap_end:
            candidates.append((offset + position, word, absolute_end))
    return candidates