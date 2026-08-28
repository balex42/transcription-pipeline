import numpy as np

from speech_transcriber.models import ASRWord, AudioSegment
from speech_transcriber.transcription.segments import reconcile_segment_words


def segments() -> list[AudioSegment]:
    return [
        AudioSegment(0, 0, 10, np.array([], dtype=np.float32)),
        AudioSegment(1, 8, 18, np.array([], dtype=np.float32)),
        AudioSegment(2, 16, 23, np.array([], dtype=np.float32)),
    ]


def test_reconciler_uses_half_overlap_ownership_without_duplicates() -> None:
    merged = reconcile_segment_words(
        segments(),
        {
            0: [ASRWord("first", 3), ASRWord("before", 8.9), ASRWord("late", 9.2)],
            1: [ASRWord("early", 0.8), ASRWord("middle", 5), ASRWord("late", 9.1)],
            2: [ASRWord("late", 1.1), ASRWord("final", 7)],
        },
    )
    assert [(word.text, word.end) for word in merged] == [
        ("first", 3),
        ("before", 8.9),
        ("middle", 13),
        ("late", 17.1),
        ("final", 23),
    ]


def test_exact_half_overlap_boundary_belongs_to_later_chunk() -> None:
    merged = reconcile_segment_words(
        segments()[:2],
        {0: [ASRWord("old", 9)], 1: [ASRWord("new", 1)]},
    )
    assert [(word.text, word.end) for word in merged] == [("new", 9)]


def test_final_endpoint_is_retained() -> None:
    merged = reconcile_segment_words(segments(), {2: [ASRWord("end", 7)]})
    assert [(word.text, word.end) for word in merged] == [("end", 23)]
