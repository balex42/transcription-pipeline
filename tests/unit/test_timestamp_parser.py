import pytest

from meeting_transcriber.errors import TimestampParseError
from meeting_transcriber.transcription.timestamp_parser import parse_timestamped_words


def test_parses_simple_sequence_with_whitespace() -> None:
    words = parse_timestamped_words("  guten [T:45]\n morgen   [T:82]  ", chunk_id=3)
    assert [(word.text, word.end, word.chunk_id) for word in words] == [
        ("guten", 0.45, 3),
        ("morgen", 0.82, 3),
    ]


def test_unwraps_ten_second_rollover() -> None:
    words = parse_timestamped_words("a [T:950] b [T:990] c [T:20] d [T:85]")
    assert [word.end for word in words] == [9.5, 9.9, 10.2, 10.85]


def test_unwraps_multiple_rollovers() -> None:
    words = parse_timestamped_words("a [T:900] b [T:10] c [T:5] d [T:999] e [T:4]")
    assert [word.end for word in words] == pytest.approx([9.0, 10.1, 20.05, 29.99, 30.04])


def test_silence_is_not_a_word_but_provides_boundary() -> None:
    words = parse_timestamped_words("hallo [T:20] _ [T:70] welt [T:95]")
    assert [word.text for word in words] == ["hallo", "welt"]
    assert words[1].previous_boundary == 0.7


@pytest.mark.parametrize("text", ["hello [T:abc]", "hello [T:1000]", "hello [T:20] stray"])
def test_rejects_malformed_timestamp_output(text: str) -> None:
    with pytest.raises(TimestampParseError):
        parse_timestamped_words(text)


def test_timestamps_are_monotonic_after_unwrap() -> None:
    words = parse_timestamped_words("a [T:999] b [T:0] c [T:0]")
    assert [word.end for word in words] == sorted(word.end for word in words)
