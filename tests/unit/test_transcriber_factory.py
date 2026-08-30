from pathlib import Path

import pytest

from speech_transcriber.config import (
    DEFAULT_CANARY_MODEL,
    DEFAULT_FASTER_WHISPER_MODEL,
    DEFAULT_GRANITE_MODEL,
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_PRIMELINE_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_VOXTRAL_DELAY_MS,
    DEFAULT_VOXTRAL_MODEL,
    DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS,
    PipelineConfig,
)
from speech_transcriber.errors import UnsupportedASRBackendError
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.canary import CanaryTranscriber
from speech_transcriber.transcription.factory import create_transcriber
from speech_transcriber.transcription.faster_whisper import FasterWhisperTranscriber
from speech_transcriber.transcription.granite import GraniteTranscriber
from speech_transcriber.transcription.nemotron import NemotronTranscriber
from speech_transcriber.transcription.parakeet import ParakeetTranscriber
from speech_transcriber.transcription.primeline import PrimelineTranscriber
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
        ("primeline", PrimelineTranscriber, DEFAULT_PRIMELINE_MODEL),
        ("qwen", QwenTranscriber, DEFAULT_QWEN_MODEL),
        ("nemotron", NemotronTranscriber, DEFAULT_NEMOTRON_MODEL),
        ("voxtral", VoxtralTranscriber, DEFAULT_VOXTRAL_MODEL),
        ("faster-whisper", FasterWhisperTranscriber, DEFAULT_FASTER_WHISPER_MODEL),
        ("canary", CanaryTranscriber, DEFAULT_CANARY_MODEL),
        ("granite", GraniteTranscriber, DEFAULT_GRANITE_MODEL),
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
    if backend == "faster-whisper":
        assert transcriber.capabilities == TranscriberCapabilities(True, True, True, True)
        assert transcriber.compute_type == "float16"
        assert transcriber.language == "de-DE"
    if backend == "canary":
        assert transcriber.capabilities == TranscriberCapabilities(True, True, True, True)
        assert transcriber.source_language == "de"
        assert transcriber.target_language == "de"
        assert transcriber.chunk_duration_seconds == 10.0
    if backend == "primeline":
        assert transcriber.capabilities == TranscriberCapabilities(True, True, True, True)
        assert transcriber.capabilities.requires_forced_alignment is False
    if backend == "granite":
        assert transcriber.capabilities == TranscriberCapabilities(
            word_start_timestamps=False,
            word_end_timestamps=True,
            punctuation=False,
            capitalization=False,
        )
        assert transcriber.segment_duration == 180.0
        assert transcriber.segment_overlap == 15.0


def test_granite_segment_settings_from_config_are_passed_to_adapter() -> None:
    transcriber = create_transcriber(
        PipelineConfig(
            Path("in.wav"),
            Path("out"),
            Path("work"),
            asr_backend="granite",
            granite_segment_duration=90.0,
            granite_segment_overlap=10.0,
        ),
        "cpu",
    )
    assert transcriber.segment_duration == 90.0
    assert transcriber.segment_overlap == 10.0


def test_granite_imports_stay_lazy_without_transformers_or_torch() -> None:
    """Factory construction must not import Transformers, Torch, or PEFT."""
    command = """
import builtins, sys

blocked = ("transformers", "torch", "peft")
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise ImportError(f"blocked import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from pathlib import Path
from speech_transcriber.config import PipelineConfig
from speech_transcriber.transcription.factory import create_transcriber
config = PipelineConfig(Path("in.wav"), Path("out"), Path("work"), asr_backend="granite")
transcriber = create_transcriber(config, "cpu")
assert type(transcriber).__name__ == "GraniteTranscriber"
assert transcriber.model_reference == "ibm-granite/granite-speech-4.1-2b-plus"
print("ok")
"""
    import subprocess
    import sys

    completed = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")


def test_primeline_does_not_reuse_the_transformers_parakeet_adapter() -> None:
    transcriber = create_transcriber(config("primeline"), "cpu")

    assert type(transcriber) is PrimelineTranscriber
    assert not isinstance(transcriber, ParakeetTranscriber)
    assert transcriber.model_reference == DEFAULT_PRIMELINE_MODEL


def test_canary_chunk_duration_from_config_is_passed_to_adapter() -> None:
    transcriber = create_transcriber(
        PipelineConfig(
            Path("in.wav"),
            Path("out"),
            Path("work"),
            asr_backend="canary",
            canary_chunk_duration_seconds=15.0,
        ),
        "cuda",
    )
    assert transcriber.chunk_duration_seconds == 15.0
    assert transcriber.working_directory == Path("work")


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


def test_faster_whisper_compute_type_from_config_is_passed_to_adapter() -> None:
    transcriber = create_transcriber(
        PipelineConfig(
            Path("in.wav"),
            Path("out"),
            Path("work"),
            asr_backend="faster-whisper",
            faster_whisper_compute_type="bfloat16",
        ),
        "cuda",
    )
    assert transcriber.compute_type == "bfloat16"
    assert transcriber.dtype_name == "bfloat16"


def test_explicit_model_overrides_backend_default() -> None:
    transcriber = create_transcriber(config("qwen", "/models/qwen"), "cpu")
    assert transcriber.model_reference == "/models/qwen"


@pytest.mark.parametrize("backend", ["invalid", "granite-speech"])
def test_invalid_backend_fails_clearly(backend: str) -> None:
    invalid = object.__new__(PipelineConfig)
    object.__setattr__(invalid, "asr_backend", backend)
    object.__setattr__(invalid, "asr_model", None)
    with pytest.raises(UnsupportedASRBackendError, match=backend):
        create_transcriber(invalid, "cpu")
