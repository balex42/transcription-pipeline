from pathlib import Path

import pytest

from speech_transcriber.config import (
    DEFAULT_CANARY_CHUNK_DURATION_SECONDS,
    DEFAULT_CANARY_MODEL,
    DEFAULT_FASTER_WHISPER_COMPUTE_TYPE,
    DEFAULT_FASTER_WHISPER_MODEL,
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_PRIMELINE_MODEL,
    DEFAULT_QWEN_ALIGNER_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_VOXTRAL_DELAY_MS,
    DEFAULT_VOXTRAL_MODEL,
    DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS,
    PipelineConfig,
)


def make(overrides: dict[str, object], env: dict[str, str] | None = None) -> PipelineConfig:
    return PipelineConfig.from_environment(Path("in.wav"), Path("out"), overrides, env or {})


def test_config_default_backend_is_parakeet_internally_only() -> None:
    """The internal default exists for tests/helpers; the recognize CLI requires --backend."""
    config = make({})
    assert (config.asr_backend, config.resolved_asr_model) == ("parakeet", DEFAULT_PARAKEET_MODEL)
    assert (config.parakeet_segment_duration, config.parakeet_segment_overlap) == (180.0, 15.0)

    from speech_transcriber import cli

    recognize = cli.subcommands(cli.build_parser())["recognize"]
    backend = next(
        action for action in recognize._actions if action.dest == "backend"  # noqa: SLF001
    )
    assert backend.required and backend.default is None


def test_environment_backend_and_default_model_mapping() -> None:
    assert make({}, {"ASR_BACKEND": "nemotron"}).resolved_asr_model == DEFAULT_NEMOTRON_MODEL
    assert make({}, {"ASR_BACKEND": "voxtral"}).resolved_asr_model == DEFAULT_VOXTRAL_MODEL
    assert (
        make({}, {"ASR_BACKEND": "faster-whisper"}).resolved_asr_model
        == DEFAULT_FASTER_WHISPER_MODEL
    )
    assert make({}, {"ASR_BACKEND": "canary"}).resolved_asr_model == DEFAULT_CANARY_MODEL
    assert make({}, {"ASR_BACKEND": "primeline"}).resolved_asr_model == DEFAULT_PRIMELINE_MODEL


def test_primeline_uses_the_primeline_repository_by_default() -> None:
    config = make({"asr_backend": "primeline"})
    assert config.asr_backend == "primeline"
    assert config.resolved_asr_model == "primeline/parakeet-primeline"


def test_primeline_explicit_asr_model_override_wins() -> None:
    config = make({"asr_backend": "primeline"}, {"ASR_MODEL": "/models/primeline"})
    assert config.resolved_asr_model == "/models/primeline"


def test_faster_whisper_uses_float16_compute_type_by_default() -> None:
    config = make({"asr_backend": "faster-whisper"})
    assert config.faster_whisper_compute_type == DEFAULT_FASTER_WHISPER_COMPUTE_TYPE


def test_canary_uses_ten_second_chunks_by_default() -> None:
    config = make({"asr_backend": "canary"})
    assert config.canary_chunk_duration_seconds == DEFAULT_CANARY_CHUNK_DURATION_SECONDS


def test_canary_chunk_duration_allows_an_override() -> None:
    assert make({}, {"CANARY_CHUNK_DURATION": "20"}).canary_chunk_duration_seconds == 20.0
    assert make({"canary_chunk_duration_seconds": 15.5}).canary_chunk_duration_seconds == 15.5


@pytest.mark.parametrize("duration", [0, -5])
def test_canary_rejects_nonpositive_chunk_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="canary_chunk_duration_seconds"):
        make({"asr_backend": "canary", "canary_chunk_duration_seconds": duration})


def test_faster_whisper_compute_type_allows_an_override() -> None:
    assert (
        make(
            {"asr_backend": "faster-whisper"},
            {"FASTER_WHISPER_COMPUTE_TYPE": "int8_float16"},
        ).faster_whisper_compute_type
        == "int8_float16"
    )


@pytest.mark.parametrize("compute_type", ["float8", "int4", ""])
def test_faster_whisper_rejects_unsupported_compute_type(compute_type: str) -> None:
    with pytest.raises(ValueError, match="faster_whisper_compute_type"):
        make({"asr_backend": "faster-whisper", "faster_whisper_compute_type": compute_type})


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
