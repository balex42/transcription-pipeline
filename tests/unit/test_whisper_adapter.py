import pytest

from meeting_transcriber.errors import ASROutputError
from meeting_transcriber.transcription.whisper import normalize_whisper_chunks


def test_normalizes_word_timestamps_and_preserves_punctuation() -> None:
    words = normalize_whisper_chunks(
        [
            {"text": " Guten", "timestamp": (0.0, 0.4)},
            {"text": " Morgen!", "timestamp": (0.4, 0.9)},
        ],
    )
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Guten", 0.0, 0.4),
        ("Morgen!", 0.4, 0.9),
    ]


def test_empty_output_and_malformed_timestamp() -> None:
    assert normalize_whisper_chunks([]) == []
    with pytest.raises(ASROutputError):
        normalize_whisper_chunks([{"text": "Hallo", "timestamp": (0,)}])
