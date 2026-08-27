from pathlib import Path

import pytest

from meeting_transcriber.config import (
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_QWEN_ALIGNER_MODEL,
    DEFAULT_QWEN_MODEL,
    PipelineConfig,
)


def make(overrides: dict[str, object], env: dict[str, str] | None = None) -> PipelineConfig:
    return PipelineConfig.from_environment(Path("in.wav"), Path("out"), overrides, env or {})


def test_default_backend_is_parakeet() -> None:
    config = make({})
    assert (config.asr_backend, config.resolved_asr_model) == ("parakeet", DEFAULT_PARAKEET_MODEL)
    assert (config.parakeet_segment_duration, config.parakeet_segment_overlap) == (180.0, 15.0)


def test_environment_backend_and_default_model_mapping() -> None:
    assert make({}, {"ASR_BACKEND": "nemotron"}).resolved_asr_model == DEFAULT_NEMOTRON_MODEL


def test_cli_backend_and_model_override_environment() -> None:
    config = make(
        {"asr_backend": "qwen", "asr_model": "/models/qwen"},
        {"ASR_BACKEND": "parakeet", "ASR_MODEL": "/models/parakeet"},
    )
    assert (config.asr_backend, config.resolved_asr_model) == ("qwen", "/models/qwen")


def test_qwen_uses_asr_and_forced_aligner_defaults() -> None:
    config = make({"asr_backend": "qwen"})
    assert config.resolved_asr_model == DEFAULT_QWEN_MODEL
    assert config.qwen_aligner_model == DEFAULT_QWEN_ALIGNER_MODEL
    assert (config.qwen_segment_duration, config.qwen_segment_overlap) == (240.0, 15.0)


def test_environment_segment_settings_override_qwen_defaults() -> None:
    config = make(
        {"asr_backend": "qwen"},
        {"QWEN_SEGMENT_DURATION": "120", "QWEN_SEGMENT_OVERLAP": "20"},
    )
    assert (config.qwen_segment_duration, config.qwen_segment_overlap) == (120.0, 20.0)


def test_explicit_segment_settings_override_qwen_defaults() -> None:
    config = make(
        {
            "asr_backend": "qwen",
            "qwen_segment_duration": 120,
            "qwen_segment_overlap": 20,
            "qwen_aligner_model": "/models/aligner",
        },
        {
            "ASR_MODEL": "/models/qwen",
            "QWEN_SEGMENT_DURATION": "60",
            "QWEN_SEGMENT_OVERLAP": "5",
            "QWEN_ALIGNER_MODEL": "/models/environment-aligner",
        },
    )
    assert (config.qwen_segment_duration, config.qwen_segment_overlap) == (120.0, 20.0)
    assert (config.resolved_asr_model, config.qwen_aligner_model) == (
        "/models/qwen",
        "/models/aligner",
    )


def test_qwen_rejects_segments_above_forced_alignment_limit() -> None:
    with pytest.raises(ValueError, match="forced-aligner limit"):
        make({"asr_backend": "qwen", "qwen_segment_duration": 301})


def test_obsolete_granite_configuration_fails_clearly() -> None:
    with pytest.raises(ValueError, match="GRANITE_MODEL is no longer supported"):
        make({}, {"GRANITE_MODEL": "/models/granite"})
