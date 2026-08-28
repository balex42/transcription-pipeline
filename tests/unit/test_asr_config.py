from pathlib import Path

import pytest

from speech_transcriber.config import (
    DEFAULT_COHERE_MODEL,
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_QWEN_ALIGNER_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_VOXTRAL_DELAY_MS,
    DEFAULT_VOXTRAL_MODEL,
    DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS,
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
    assert make({}, {"ASR_BACKEND": "voxtral"}).resolved_asr_model == DEFAULT_VOXTRAL_MODEL
    assert make({}, {"ASR_BACKEND": "cohere"}).resolved_asr_model == DEFAULT_COHERE_MODEL


def test_nemotron_uses_highest_accuracy_lookahead_by_default() -> None:
    assert make({}).nemotron_num_lookahead_tokens == DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS


def test_nemotron_lookahead_allows_a_low_latency_override() -> None:
    assert make({}, {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "0"}).nemotron_num_lookahead_tokens == 0


def test_voxtral_uses_highest_accuracy_delay_by_default() -> None:
    assert make({}).voxtral_delay_ms == DEFAULT_VOXTRAL_DELAY_MS


def test_voxtral_delay_allows_a_lower_latency_override() -> None:
    assert make({}, {"VOXTRAL_DELAY_MS": "480"}).voxtral_delay_ms == 480


def test_voxtral_uses_calibrated_timestamp_offset_by_default() -> None:
    assert make({}).voxtral_timestamp_offset_tokens == DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS


def test_voxtral_timestamp_offset_allows_an_override() -> None:
    assert make({}, {"VOXTRAL_TIMESTAMP_OFFSET_TOKENS": "6"}).voxtral_timestamp_offset_tokens == 6


def test_voxtral_rejects_delay_outside_supported_range() -> None:
    with pytest.raises(ValueError, match="Voxtral delay must be between"):
        make({}, {"VOXTRAL_DELAY_MS": "2401"})


def test_voxtral_rejects_delay_that_is_not_a_multiple_of_80ms() -> None:
    with pytest.raises(ValueError, match="multiple of 80ms"):
        make({}, {"VOXTRAL_DELAY_MS": "500"})


def test_voxtral_rejects_unsupported_delay_between_1200ms_and_2400ms() -> None:
    with pytest.raises(ValueError, match="up to 1200ms, or 2400ms"):
        make({}, {"VOXTRAL_DELAY_MS": "1280"})


@pytest.mark.parametrize("offset", ["-1", "31"])
def test_voxtral_rejects_timestamp_offset_outside_delay_horizon(offset: str) -> None:
    with pytest.raises(ValueError, match="timestamp offset"):
        make({}, {"VOXTRAL_TIMESTAMP_OFFSET_TOKENS": offset})


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


def test_cohere_uses_forced_alignment_segment_defaults() -> None:
    config = make({"asr_backend": "cohere"})
    assert config.resolved_asr_model == DEFAULT_COHERE_MODEL
    assert (config.cohere_segment_duration, config.cohere_segment_overlap) == (30.0, 5.0)


def test_cohere_allows_a_supported_forced_alignment_language() -> None:
    assert make({"asr_backend": "cohere", "language": "ja-JP"}).language == "ja-JP"


@pytest.mark.parametrize("language", ["ar", "el-GR", "nl-NL", "pl", "vi-VN"])
def test_cohere_rejects_language_not_supported_by_the_forced_aligner(language: str) -> None:
    with pytest.raises(ValueError, match="requires word timestamps"):
        make({"asr_backend": "cohere", "language": language})


def test_obsolete_granite_configuration_fails_clearly() -> None:
    with pytest.raises(ValueError, match="GRANITE_MODEL is no longer supported"):
        make({}, {"GRANITE_MODEL": "/models/granite"})
