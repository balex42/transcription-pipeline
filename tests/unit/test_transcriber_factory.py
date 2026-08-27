from pathlib import Path

import pytest

from meeting_transcriber.config import (
    DEFAULT_GRANITE_MODEL,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_WHISPER_MODEL,
    PipelineConfig,
)
from meeting_transcriber.errors import UnsupportedASRBackendError
from meeting_transcriber.transcription.factory import create_transcriber
from meeting_transcriber.transcription.granite import GraniteTranscriber
from meeting_transcriber.transcription.parakeet import ParakeetTranscriber
from meeting_transcriber.transcription.whisper import WhisperTranscriber


def config(backend: str, model: str | None = None) -> PipelineConfig:
    return PipelineConfig(
        Path("in.wav"), Path("out"), Path("work"), asr_backend=backend, asr_model=model
    )


@pytest.mark.parametrize(
    ("backend", "adapter", "model"),
    [
        ("parakeet", ParakeetTranscriber, DEFAULT_PARAKEET_MODEL),
        ("whisper", WhisperTranscriber, DEFAULT_WHISPER_MODEL),
        ("granite", GraniteTranscriber, DEFAULT_GRANITE_MODEL),
    ],
)
def test_factory_selects_adapter_and_default_model(
    backend: str, adapter: type[object], model: str
) -> None:
    transcriber = create_transcriber(config(backend), "cpu")
    assert isinstance(transcriber, adapter)
    assert transcriber.model_reference == model


def test_explicit_model_overrides_backend_default() -> None:
    transcriber = create_transcriber(config("whisper", "/models/whisper"), "cpu")
    assert transcriber.model_reference == "/models/whisper"


def test_invalid_backend_fails_clearly() -> None:
    invalid = object.__new__(PipelineConfig)
    object.__setattr__(invalid, "asr_backend", "invalid")
    object.__setattr__(invalid, "asr_model", None)
    with pytest.raises(UnsupportedASRBackendError, match="invalid"):
        create_transcriber(invalid, "cpu")
