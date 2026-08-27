import pytest

from meeting_transcriber.errors import ASROutputError
from meeting_transcriber.transcription.whisper import normalize_whisper_text


def test_normalize_whisper_text_trims_plain_transcript() -> None:
    assert normalize_whisper_text({"text": " Guten Morgen "}) == "Guten Morgen"


@pytest.mark.parametrize("result", [None, {}, {"text": None}, {"text": ["Hallo"]}])
def test_normalize_whisper_text_rejects_malformed_responses(result: object) -> None:
    with pytest.raises(ASROutputError):
        normalize_whisper_text(result)
