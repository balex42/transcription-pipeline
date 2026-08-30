from pathlib import Path

import pytest

from speech_transcriber.config import (
    DEFAULT_CANARY_MODEL,
    DEFAULT_FASTER_WHISPER_MODEL,
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_PRIMELINE_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_VOXTRAL_DELAY_MS,
    DEFAULT_VOXTRAL_MODEL,
    DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS,
    RecognitionConfig,
)
from speech_transcriber.errors import UnsupportedASRBackendError
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.canary import CanaryTranscriber
from speech_transcriber.transcription.factory import create_transcriber
from speech_transcriber.transcription.faster_whisper import FasterWhisperTranscriber
from speech_transcriber.transcription.nemotron import NemotronTranscriber
from speech_transcriber.transcription.parakeet import ParakeetTranscriber
from speech_transcriber.transcription.primeline import PrimelineTranscriber
from speech_transcriber.transcription.qwen import QwenTranscriber
from speech_transcriber.transcription.voxtral import VoxtralTranscriber


def config(
    backend: str, model: str | None = None
) -> RecognitionConfig:
    return RecognitionConfig(
        prepared_path=Path("prepared"),
        output_directory=Path("out"),
        working_directory=Path("work"),
        asr_backend=backend,
        asr_model=model,
    )


@pytest.mark.parametrize(
    ("backend", "adapter", "model"),
    [
        ("parakeet", ParakeetTranscriber, DEFAULT_PARAKEET_MODEL),
        ("primeline", PrimelineTranscriber, DEFAULT_PRIMELINE_MODEL),
        ("qwen", QwenTranscriber, DEFAULT_QWEN_MODEL),
        ("nemotron", NemotronTranscriber, DEFAULT_NEMOTRON_MODEL),
        ("voxtral", VoxtralTranscriber, DEFAULT_VOXTRAL_MODEL),
        ("faster-whisper", FasterWhisperTranscriber, DEFAULT_FASTER_WHISPER_MODEL),
        ("canary", CanaryTranscriber, DEFAULT_CANARY_MODEL),
    ],
)
def test_factory_selects_adapter_and_default_model(
    backend: str, adapter: type[object], model: str
) -> None:
    transcriber = create_transcriber(config(backend), "cpu", "de-DE")
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
    if backend == "faster-whisper":
        assert transcriber.capabilities == TranscriberCapabilities(True, True, True, True)
        assert transcriber.compute_type == "float16"
        assert transcriber.language == "de-DE"
    if backend == "canary":
        assert transcriber.chunk_duration_seconds == 10.0
    if backend == "primeline":
        assert transcriber.capabilities == TranscriberCapabilities(True, True, True, True)
        assert transcriber.capabilities.requires_forced_alignment is False


def test_primeline_does_not_reuse_the_transformers_parakeet_adapter() -> None:
    transcriber = create_transcriber(config("primeline"), "cpu", "de-DE")

    assert type(transcriber) is PrimelineTranscriber
    assert not isinstance(transcriber, ParakeetTranscriber)
    assert transcriber.model_reference == DEFAULT_PRIMELINE_MODEL


def test_prepared_language_is_passed_through_to_language_conditioned_adapters() -> None:
    """fr-FR from the prepared artifact reaches Qwen, Nemotron, and Canary unchanged."""
    from speech_transcriber.transcription.qwen.recognizer import QwenRecognizer

    qwen = create_transcriber(config("qwen"), "cpu", "fr-FR")
    assert isinstance(qwen._recognizer, QwenRecognizer)  # noqa: SLF001
    assert qwen._recognizer.language == "fr-FR"  # noqa: SLF001
    nemotron = create_transcriber(config("nemotron"), "cpu", "en-US")
    assert nemotron.language == "en-US"
    canary = create_transcriber(config("canary"), "cpu", "fr-FR")
    assert canary.requested_language == "fr-FR"


def test_faster_whisper_receives_the_prepared_language_unchanged() -> None:
    transcriber = create_transcriber(config("faster-whisper"), "cpu", "en-US")
    assert transcriber.language == "en-US"


def test_canary_chunk_duration_from_config_is_passed_to_adapter() -> None:
    transcriber = create_transcriber(
        RecognitionConfig(
            prepared_path=Path("prepared"),
            output_directory=Path("out"),
            working_directory=Path("work"),
            asr_backend="canary",
            canary_chunk_duration_seconds=15.0,
        ),
        "cuda",
        "de-DE",
    )
    assert transcriber.chunk_duration_seconds == 15.0
    assert transcriber.working_directory == Path("work")


def test_voxtral_delay_from_config_is_passed_to_adapter() -> None:
    transcriber = create_transcriber(
        RecognitionConfig(
            prepared_path=Path("prepared"),
            output_directory=Path("out"),
            working_directory=Path("work"),
            asr_backend="voxtral",
            voxtral_delay_ms=1200,
            voxtral_timestamp_offset_tokens=6,
        ),
        "cpu",
        "de-DE",
    )
    assert transcriber.delay_ms == 1200
    assert transcriber.timestamp_offset_tokens == 6


def test_faster_whisper_compute_type_from_config_is_passed_to_adapter() -> None:
    transcriber = create_transcriber(
        RecognitionConfig(
            prepared_path=Path("prepared"),
            output_directory=Path("out"),
            working_directory=Path("work"),
            asr_backend="faster-whisper",
            faster_whisper_compute_type="bfloat16",
        ),
        "cuda",
        "de-DE",
    )
    assert transcriber.compute_type == "bfloat16"
    assert transcriber.dtype_name == "bfloat16"


def test_explicit_model_overrides_backend_default() -> None:
    transcriber = create_transcriber(config("qwen", "/models/qwen"), "cpu", "de-DE")
    assert transcriber.model_reference == "/models/qwen"


@pytest.mark.parametrize("language", [None, ""])
def test_factory_does_not_manufacture_a_default_language(language: object) -> None:
    with pytest.raises(ValueError, match="prepared language"):
        create_transcriber(config("qwen"), "cpu", language)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["invalid", "unknown-asr"])
def test_invalid_backend_fails_clearly(backend: str) -> None:
    invalid = object.__new__(RecognitionConfig)
    object.__setattr__(invalid, "asr_backend", backend)
    object.__setattr__(invalid, "asr_model", None)
    with pytest.raises(UnsupportedASRBackendError, match=backend):
        create_transcriber(invalid, "cpu", "de-DE")
