from speech_transcriber.alignment.speaker import UNKNOWN_SPEAKER, OverlapSpeakerAligner
from speech_transcriber.models import ASRWord, DiarizationSegment


def test_one_speaker() -> None:
    result = OverlapSpeakerAligner().align(
        [ASRWord("hallo", 1.0)], [DiarizationSegment("SPEAKER_00", 0, 2)]
    )
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].start_is_inferred


def test_largest_overlap_wins_when_boundary_inside_word() -> None:
    result = OverlapSpeakerAligner().align(
        [ASRWord("wechsel", 2.0, start=1.0)],
        [DiarizationSegment("SPEAKER_00", 0, 1.4), DiarizationSegment("SPEAKER_01", 1.4, 3)],
    )
    assert result[0].speaker == "SPEAKER_01"


def test_timestamp_tolerance_and_unknown_fallback() -> None:
    aligner = OverlapSpeakerAligner(tolerance_seconds=0.25)
    near = aligner.align(
        [ASRWord("nah", 1.1, start=1.1)], [DiarizationSegment("SPEAKER_00", 0, 1)]
    )
    missing = aligner.align(
        [ASRWord("fern", 5, start=5)], [DiarizationSegment("SPEAKER_00", 0, 1)]
    )
    assert near[0].speaker == "SPEAKER_00"
    assert missing[0].speaker == UNKNOWN_SPEAKER


def test_multiple_speakers_and_missing_coverage() -> None:
    words = [ASRWord("eins", 0.8), ASRWord("zwei", 2.1)]
    segments = [DiarizationSegment("SPEAKER_00", 0, 1), DiarizationSegment("SPEAKER_01", 1.5, 3)]
    assert [word.speaker for word in OverlapSpeakerAligner().align(words, segments)] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]


def test_end_only_voxtral_words_infer_starts_from_the_preceding_boundary() -> None:
    result = OverlapSpeakerAligner().align(
        [ASRWord("eins", 1.0), ASRWord("zwei", 2.0)],
        [DiarizationSegment("SPEAKER_00", 0, 1.4), DiarizationSegment("SPEAKER_01", 1.4, 3)],
    )
    assert [(word.start, word.end, word.speaker, word.start_is_inferred) for word in result] == [
        (0.0, 1.0, "SPEAKER_00", True),
        (1.0, 2.0, "SPEAKER_01", True),
    ]
