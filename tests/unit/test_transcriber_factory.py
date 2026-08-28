from pathlib import Path

import pytest

from speech_transcriber.config import (
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_VOXTRAL_DELAY_MS,
    DEFAULT_VOXTRAL_MODEL,
    DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS,
    PipelineConfig,
)
from speech_transcriber.errors import UnsupportedASRBackendError
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.factory import create_transcriber
from speech_transcriber.transcription.nemotron import NemotronTranscriber
from speech_transcriber.transcription.parakeet import ParakeetTranscriber
from speech_transcriber.transcription.qwen import QwenTranscriber
from speech_transcriber.transcription.voxtral import VoxtralTranscriber


def config(backend: str, model: str | None = None) -> PipelineConfig:
    return PipelineConfig(
        Path("in.wav"), Path("out"), Path("work"), asr_backend=backend, asr_model=model
    )


@pytest.mark.parametrize(
    ("backend", "adapter", "model"),
    [
        ("parakeet", ParakeetTranscriber, DEFAULT_PARAKEET_MODEL),
        ("qwen", QwenTranscriber, DEFAULT_QWEN_MODEL),
        ("nemotron", NemotronTranscriber, DEFAULT_NEMOTRON_MODEL),
        ("voxtral", VoxtralTranscriber, DEFAULT_VOXTRAL_MODEL),
    ],
)
def test_factory_selects_adapter_and_default_model(
    backend: str, adapter: type[object], model: str
) -> None:
    transcriber = create_transcriber(config(backend), "cpu")
    assert isinstance(transcriber, adapter)
    assert transcriber.model_reference == model
    if backend == "qwen":
        assert transcriber.capabilities.requires_forced_alignment is True
    if backend == "nemotron":
        assert transcriber.capabilities.streaming is True
    if backend == "voxtral":
        assert transcriber.capabilities == TranscriberCapabilities(
            False, True, True, True, streaming=True
        )
        assert transcriber.delay_ms == DEFAULT_VOXTRAL_DELAY_MS
        assert transcriber.timestamp_offset_tokens == DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS


def test_voxtral_delay_from_config_is_passed_to_adapter() -> None:
    transcriber = create_transcriber(
        PipelineConfig(
            Path("in.wav"),
            Path("out"),
            Path("work"),
            asr_backend="voxtral",
            voxtral_delay_ms=1200,
            voxtral_timestamp_offset_tokens=6,
        ),
        "cpu",
    )
    assert transcriber.delay_ms == 1200
    assert transcriber.timestamp_offset_tokens == 6


def test_explicit_model_overrides_backend_default() -> None:
    transcriber = create_transcriber(config("qwen", "/models/qwen"), "cpu")
    assert transcriber.model_reference == "/models/qwen"


@pytest.mark.parametrize("backend", ["invalid", "granite"])
def test_invalid_backend_fails_clearly(backend: str) -> None:
    invalid = object.__new__(PipelineConfig)
    object.__setattr__(invalid, "asr_backend", backend)
    object.__setattr__(invalid, "asr_model", None)
    with pytest.raises(UnsupportedASRBackendError, match=backend):
        create_transcriber(invalid, "cpu")
