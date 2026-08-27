from pathlib import Path

from meeting_transcriber.config import (
    DEFAULT_GRANITE_MODEL,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_WHISPER_MODEL,
    PipelineConfig,
)


def make(overrides: dict[str, object], env: dict[str, str] | None = None) -> PipelineConfig:
    return PipelineConfig.from_environment(Path("in.wav"), Path("out"), overrides, env or {})


def test_default_backend_is_parakeet() -> None:
    config = make({})
    assert (config.asr_backend, config.resolved_asr_model) == ("parakeet", DEFAULT_PARAKEET_MODEL)


def test_environment_backend_and_default_model_mapping() -> None:
    assert make({}, {"ASR_BACKEND": "whisper"}).resolved_asr_model == DEFAULT_WHISPER_MODEL


def test_cli_backend_and_model_override_environment() -> None:
    config = make(
        {"asr_backend": "whisper", "asr_model": "/models/whisper"},
        {"ASR_BACKEND": "parakeet", "ASR_MODEL": "/models/parakeet"},
    )
    assert (config.asr_backend, config.resolved_asr_model) == ("whisper", "/models/whisper")


def test_granite_model_environment_remains_compatible() -> None:
    config = make({"asr_backend": "granite"}, {"GRANITE_MODEL": "/models/granite"})
    assert config.resolved_asr_model == "/models/granite"
    assert DEFAULT_GRANITE_MODEL
