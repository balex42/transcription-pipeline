from speech_transcriber.models import AttributedWord
from speech_transcriber.turns.builder import TurnBuilder


def word(text: str, start: float, end: float, speaker: str) -> AttributedWord:
    return AttributedWord(text, start, end, speaker)


def test_groups_consecutive_same_speaker() -> None:
    turns = TurnBuilder(1).build(
        [word("guten", 0, 0.3, "SPEAKER_00"), word("morgen", 0.3, 0.8, "SPEAKER_00")]
    )
    assert [(turn.speaker, turn.text, turn.start, turn.end) for turn in turns] == [
        ("SPEAKER_00", "guten morgen", 0, 0.8)
    ]


def test_starts_new_turn_for_speaker_change_gap_and_unknown() -> None:
    turns = TurnBuilder(0.5).build(
        [
            word("a", 0, 0.1, "SPEAKER_00"),
            word("b", 0.2, 0.3, "SPEAKER_01"),
            word("c", 1, 1.1, "SPEAKER_01"),
            word("d", 1.2, 1.3, "UNKNOWN"),
        ]
    )
    assert [turn.speaker for turn in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "UNKNOWN"]
    assert [turn.start for turn in turns] == sorted(turn.start for turn in turns)
