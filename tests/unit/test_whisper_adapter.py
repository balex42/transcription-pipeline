import pytest

from meeting_transcriber.errors import ASROutputError
from meeting_transcriber.transcription.whisper import normalize_whisper_chunks


def test_normalizes_word_timestamps_and_preserves_punctuation() -> None:
    words = normalize_whisper_chunks(
        [
            {"text": " Guten", "timestamp": (0.0, 0.4)},
            {"text": " Morgen!", "timestamp": (0.4, 0.9)},
        ],
        2,
    )
    assert [(word.text, word.start, word.end, word.chunk_id) for word in words] == [
        ("Guten", 0.0, 0.4, 2),
        ("Morgen!", 0.4, 0.9, 2),
    ]


def test_empty_output_and_malformed_timestamp() -> None:
    assert normalize_whisper_chunks([], 0) == []
    with pytest.raises(ASROutputError):
        normalize_whisper_chunks([{"text": "Hallo", "timestamp": (0,)}], 0)
