import pytest

from speech_transcriber.errors import ASROutputError
from speech_transcriber.transcription.parakeet import normalize_parakeet_timestamps


def test_normalizes_word_timestamps_and_preserves_punctuation() -> None:
    words = normalize_parakeet_timestamps(
        [
            {"word": "Guten,", "start": 0.1, "end": 0.5},
            {"word": "Morgen!", "start": 0.5, "end": 1.0},
        ],
    )
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Guten,", 0.1, 0.5),
        ("Morgen!", 0.5, 1.0),
    ]


def test_accepts_nested_timestamp_bounds_and_empty_output() -> None:
    assert normalize_parakeet_timestamps([]) == []
    word = normalize_parakeet_timestamps({"words": [{"text": "Hallo.", "timestamp": (1, 2)}]})[0]
    assert (word.start, word.end, word.text) == (1.0, 2.0, "Hallo.")


def test_joins_native_timestamped_tokens_into_words() -> None:
    words = normalize_parakeet_timestamps(
        [
            [
                {"token": " Guten", "start": 0.0, "end": 0.2},
                {"token": "morgen", "start": 0.2, "end": 0.5},
                {"token": "!", "start": 0.5, "end": 0.5},
                {"token": " Zusammen", "start": 0.6, "end": 1.0},
            ]
        ],
    )
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Gutenmorgen!", 0.0, 0.5),
        ("Zusammen", 0.6, 1.0),
    ]


@pytest.mark.parametrize("timestamps", [[{"word": "bad", "start": 2, "end": 1}], "bad"])
def test_rejects_malformed_timestamps(timestamps: object) -> None:
    with pytest.raises(ASROutputError):
        normalize_parakeet_timestamps(timestamps)
