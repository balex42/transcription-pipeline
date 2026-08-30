"""Recognition configuration: backend-scoped environment parsing and validation."""

from pathlib import Path

import pytest

from speech_transcriber.config import (
    DEFAULT_CANARY_CHUNK_DURATION_SECONDS,
    DEFAULT_CANARY_MODEL,
    DEFAULT_FASTER_WHISPER_COMPUTE_TYPE,
    DEFAULT_FASTER_WHISPER_MODEL,
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_PRIMELINE_MODEL,
    DEFAULT_QWEN_ALIGNER_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_VOXTRAL_MODEL,
    FinalizationConfig,
    PreparationConfig,
    RecognitionConfig,
)
from speech_transcriber.transcription.factory import create_transcriber
from speech_transcriber.transcription.faster_whisper import FasterWhisperTranscriber


def prepare(overrides: dict[str, object], env: dict[str, str] | None = None) -> PreparationConfig:
    return PreparationConfig.from_environment(Path("in.wav"), Path("out"), overrides, env or {})


def recognize_for(
    backend: str, overrides: dict[str, object], env: dict[str, str] | None = None
) -> RecognitionConfig:
    """Build a recognition config through the CLI-shaped construction path."""
    return RecognitionConfig.from_environment(
        Path("prepared"), Path("out"), backend, {"asr_backend": backend, **overrides}, env or {}
    )


def finalize(
    overrides: dict[str, object], env: dict[str, str] | None = None
) -> FinalizationConfig:
    return FinalizationConfig.from_environment(Path("out"), overrides, env or {})


# --- Backend selection: explicit only ---------------------------------------------


def test_recognition_backend_is_a_required_constructor_value() -> None:
    """There is no implicit Parakeet fallback for asr_backend."""
    with pytest.raises(TypeError):
        RecognitionConfig(  # type: ignore[call-arg]
            prepared_path=Path("prepared"),
            output_directory=Path("out"),
            working_directory=Path("work"),
        )


def test_environment_does_not_select_the_recognition_backend() -> None:
    """ASR_BACKEND is dead: backend identity comes from the --backend argument only."""
    config = recognize_for("parakeet", {}, {"ASR_BACKEND": "nemotron"})
    assert config.asr_backend == "parakeet"
    assert config.resolved_asr_model == DEFAULT_PARAKEET_MODEL


@pytest.mark.parametrize(
    ("backend", "model"),
    [
        ("parakeet", DEFAULT_PARAKEET_MODEL),
        ("primeline", DEFAULT_PRIMELINE_MODEL),
        ("qwen", DEFAULT_QWEN_MODEL),
        ("nemotron", DEFAULT_NEMOTRON_MODEL),
        ("voxtral", DEFAULT_VOXTRAL_MODEL),
        ("faster-whisper", DEFAULT_FASTER_WHISPER_MODEL),
        ("canary", DEFAULT_CANARY_MODEL),
    ],
)
def test_explicit_backend_maps_to_its_default_model(backend: str, model: str) -> None:
    assert recognize_for(backend, {}).resolved_asr_model == model


def test_primeline_explicit_asr_model_override_wins() -> None:
    config = recognize_for("primeline", {"asr_model": "/models/primeline"}, {"ASR_MODEL": "/x"})
    assert config.resolved_asr_model == "/models/primeline"


# --- Backend-specific environment ownership ---------------------------------------


def test_parakeet_parses_its_own_segments() -> None:
    config = recognize_for("parakeet", {}, {"PARAKEET_SEGMENT_DURATION": "120"})
    assert config.parakeet_segment_duration == 120.0


def test_qwen_parses_its_own_settings() -> None:
    config = recognize_for(
        "qwen",
        {},
        {
            "QWEN_SEGMENT_DURATION": "120",
            "QWEN_SEGMENT_OVERLAP": "20",
            "QWEN_ALIGNER_MODEL": "qwen/env-aligner",
        },
    )
    assert (config.qwen_segment_duration, config.qwen_segment_overlap) == (120.0, 20.0)
    assert config.qwen_aligner_model == "qwen/env-aligner"


def test_qwen_rejects_segments_above_forced_alignment_limit() -> None:
    with pytest.raises(ValueError, match="forced-aligner limit"):
        recognize_for("qwen", {"qwen_segment_duration": 301})


@pytest.mark.parametrize(
    ("backend", "env", "field", "expected"),
    [
        ("nemotron", {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "0"}, "nemotron_num_lookahead_tokens", 0),
        ("voxtral", {"VOXTRAL_DELAY_MS": "480"}, "voxtral_delay_ms", 480),
        (
            "voxtral",
            {"VOXTRAL_TIMESTAMP_OFFSET_TOKENS": "6"},
            "voxtral_timestamp_offset_tokens",
            6,
        ),
        ("canary", {"CANARY_CHUNK_DURATION": "20"}, "canary_chunk_duration_seconds", 20.0),
        (
            "faster-whisper",
            {"FASTER_WHISPER_COMPUTE_TYPE": "int8_float16"},
            "faster_whisper_compute_type",
            "int8_float16",
        ),
    ],
)
def test_backend_specific_environment_overrides(
    backend: str, env: dict[str, str], field: str, expected: object
) -> None:
    config = recognize_for(backend, {}, env)
    assert getattr(config, field) == expected


def test_faster_whisper_uses_float16_compute_type_by_default() -> None:
    assert (
        recognize_for("faster-whisper", {}).faster_whisper_compute_type
        == DEFAULT_FASTER_WHISPER_COMPUTE_TYPE
    )


@pytest.mark.parametrize("compute_type", ["float8", "int4"])
def test_faster_whisper_rejects_unsupported_compute_type(compute_type: str) -> None:
    with pytest.raises(ValueError, match="faster_whisper_compute_type"):
        recognize_for("faster-whisper", {"faster_whisper_compute_type": compute_type})


def test_canary_rejects_nonpositive_chunk_duration() -> None:
    with pytest.raises(ValueError, match="canary_chunk_duration_seconds"):
        recognize_for("canary", {"canary_chunk_duration_seconds": 0})


def test_voxtral_rejects_invalid_streaming_settings() -> None:
    with pytest.raises(ValueError, match="Voxtral delay must be between"):
        recognize_for("voxtral", {}, {"VOXTRAL_DELAY_MS": "2401"})
    with pytest.raises(ValueError, match="multiple of 80ms"):
        recognize_for("voxtral", {}, {"VOXTRAL_DELAY_MS": "500"})
    with pytest.raises(ValueError, match="timestamp offset"):
        recognize_for("voxtral", {}, {"VOXTRAL_TIMESTAMP_OFFSET_TOKENS": "31"})


def test_parakeet_rejects_invalid_segments() -> None:
    with pytest.raises(ValueError, match="could not convert string to float"):
        recognize_for("parakeet", {"parakeet_segment_duration": "garbage"})


def test_cli_model_override_wins_over_environment_asr_model() -> None:
    config = recognize_for("qwen", {"asr_model": "/models/qwen"}, {"ASR_MODEL": "/models/x"})
    assert config.resolved_asr_model == "/models/qwen"


# --- Cross-backend environment isolation ------------------------------------------


@pytest.mark.parametrize(
    "garbage",
    [
        {"VOXTRAL_DELAY_MS": "garbage"},
        {"QWEN_SEGMENT_DURATION": "garbage"},
        {"CANARY_CHUNK_DURATION": "garbage"},
        {"FASTER_WHISPER_COMPUTE_TYPE": "float8"},
        {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "garbage"},
    ],
)
def test_parakeet_ignores_unrelated_backend_environment(garbage: dict[str, str]) -> None:
    config = recognize_for("parakeet", {}, garbage)
    assert config.asr_backend == "parakeet"
    assert (config.parakeet_segment_duration, config.parakeet_segment_overlap) == (180.0, 15.0)


@pytest.mark.parametrize(
    "garbage",
    [
        {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "garbage"},
        {"FASTER_WHISPER_COMPUTE_TYPE": "garbage"},
        {"CANARY_CHUNK_DURATION": "garbage"},
        {"PARAKEET_SEGMENT_DURATION": "garbage"},
    ],
)
def test_qwen_ignores_unrelated_backend_environment(garbage: dict[str, str]) -> None:
    config = recognize_for("qwen", {}, garbage)
    assert config.asr_backend == "qwen"
    assert (config.qwen_segment_duration, config.qwen_segment_overlap) == (240.0, 15.0)
    assert config.qwen_aligner_model == DEFAULT_QWEN_ALIGNER_MODEL


@pytest.mark.parametrize(
    "garbage",
    [
        {"QWEN_SEGMENT_DURATION": "garbage"},
        {"VOXTRAL_DELAY_MS": "garbage"},
        {"PARAKEET_SEGMENT_DURATION": "garbage"},
        {"QWEN_SEGMENT_OVERLAP": "999999"},
    ],
)
def test_canary_ignores_unrelated_backend_environment(garbage: dict[str, str]) -> None:
    config = recognize_for("canary", {}, garbage)
    assert config.asr_backend == "canary"
    assert config.canary_chunk_duration_seconds == DEFAULT_CANARY_CHUNK_DURATION_SECONDS


@pytest.mark.parametrize(
    "garbage",
    [
        {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "garbage"},
        {"QWEN_SEGMENT_DURATION": "garbage"},
        {"VOXTRAL_DELAY_MS": "garbage"},
        {"PARAKEET_SEGMENT_OVERLAP": "garbage"},
    ],
)
def test_faster_whisper_ignores_unrelated_backend_environment(garbage: dict[str, str]) -> None:
    config = recognize_for("faster-whisper", {}, garbage)
    assert config.asr_backend == "faster-whisper"
    assert config.faster_whisper_compute_type == DEFAULT_FASTER_WHISPER_COMPUTE_TYPE


@pytest.mark.parametrize(
    "garbage",
    [
        {"QWEN_SEGMENT_DURATION": "garbage"},
        {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "garbage"},
        {"VOXTRAL_DELAY_MS": "garbage"},
        {"CANARY_CHUNK_DURATION": "garbage"},
    ],
)
def test_primeline_ignores_unrelated_backend_environment(garbage: dict[str, str]) -> None:
    config = recognize_for("primeline", {}, garbage)
    assert config.asr_backend == "primeline"
    assert config.resolved_asr_model == DEFAULT_PRIMELINE_MODEL


def test_recognize_ignores_diarization_environment_garbage() -> None:
    config = recognize_for("parakeet", {}, {"PYANNOTE_MODEL": "anything"})
    assert config.device == "auto"


def test_spec_example_voxtral_garbage_does_not_break_parakeet(tmp_path: Path) -> None:
    """The exact CLI invocation from the spec survives another backend's garbage."""
    from speech_transcriber.cli import build_parser

    args = build_parser().parse_args(
        ["recognize", "--prepared", "prepared", "--backend", "parakeet", "--output", "out"]
    )
    config = RecognitionConfig.from_environment(
        args.prepared, args.output, args.backend, {},
        {"VOXTRAL_DELAY_MS": "garbage"},
    )
    assert config.asr_backend == "parakeet"


# --- Language: the prepared artifact is the only source ---------------------------


def test_recognition_config_has_no_language_field_or_override() -> None:
    config = recognize_for("parakeet", {})
    assert not hasattr(config, "language")


def test_recognition_never_parses_a_language_environment_variable() -> None:
    """LANGUAGE cannot alter recognition behavior; the artifact owns the language."""
    config = recognize_for("qwen", {}, {"LANGUAGE": "de-DE"})
    assert not hasattr(config, "language")


# --- Stage environment isolation ---------------------------------------------------


@pytest.mark.parametrize(
    "garbage",
    [
        {"VOXTRAL_DELAY_MS": "not-an-int"},
        {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "garbage"},
        {"CANARY_CHUNK_DURATION": "bad"},
        {"QWEN_SEGMENT_DURATION": "garbage"},
        {"PARAKEET_SEGMENT_DURATION": "garbage"},
        {"ASR_MODEL": "anything"},
        {"FASTER_WHISPER_COMPUTE_TYPE": "float8"},
        {"ASR_BACKEND": "nemotron"},
        {"PYANNOTE_MODEL": "anything"},
    ],
)
def test_prepare_ignores_recognition_environment_garbage(garbage: dict[str, str]) -> None:
    config = prepare({"language": "fr-FR"}, garbage)
    assert config.language == "fr-FR"
    assert config.device == "auto"


@pytest.mark.parametrize(
    "garbage",
    [
        {"PARAKEET_SEGMENT_DURATION": "garbage"},
        {"QWEN_SEGMENT_DURATION": "bad"},
        {"FASTER_WHISPER_COMPUTE_TYPE": "float8"},
        {"VOXTRAL_DELAY_MS": "not-even-words"},
        {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "ten"},
        {"NUM_SPEAKERS": "only"},
        {"PYANNOTE_MODEL": "ignored"},
        {"MIN_SPEAKERS": "x"},
        {"ASR_BACKEND": "qwen"},
    ],
)
def test_finalize_ignores_recognition_and_diarization_garbage(garbage: dict[str, str]) -> None:
    config = finalize({}, garbage)
    assert config.alignment_tolerance == 0.25
    assert config.turn_gap_seconds == 1.0
    assert not hasattr(config, "log_level")


def test_stage_configs_do_not_carry_cli_logging_state() -> None:
    assert not hasattr(prepare({}), "log_level")
    assert not hasattr(recognize_for("parakeet", {}), "log_level")
    assert not hasattr(finalize({}), "log_level")


# --- Preparation configuration -----------------------------------------------------


def test_prepare_parses_speaker_settings_and_defaults() -> None:
    config = prepare({})
    assert config.pyannote_model  # defaulted provenance is non-empty
    assert config.language == "de-DE"
    assert config.num_speakers is None


def test_prepare_applies_cli_and_environment_speaker_settings() -> None:
    from speech_transcriber.config import DEFAULT_PYANNOTE_MODEL

    config = prepare(
        {"num_speakers": 3},
        {"MIN_SPEAKERS": "1", "MAX_SPEAKERS": "5", "PYANNOTE_MODEL": "pyannote/env"},
    )
    assert (config.num_speakers, config.min_speakers, config.max_speakers) == (3, 1, 5)
    assert config.pyannote_model == "pyannote/env"
    assert DEFAULT_PYANNOTE_MODEL  # guard against accidental constant removal


def test_prepare_rejects_inverted_speaker_bounds() -> None:
    with pytest.raises(ValueError, match="min_speakers cannot exceed max_speakers"):
        prepare({"min_speakers": 5, "max_speakers": 2})


def test_prepare_rejects_nonpositive_speaker_counts() -> None:
    with pytest.raises(ValueError, match="num_speakers must be positive"):
        prepare({"num_speakers": 0})


def test_prepare_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="device"):
        prepare({"device": "tpu"})


# --- Finalization configuration -----------------------------------------------------


def test_finalize_defaults_and_overrides() -> None:
    assert (finalize({}).alignment_tolerance, finalize({}).turn_gap_seconds) == (0.25, 1.0)
    config = finalize({"alignment_tolerance": 0.5, "turn_gap_seconds": 2.0})
    assert (config.alignment_tolerance, config.turn_gap_seconds) == (0.5, 2.0)


def test_finalization_stays_backend_neutral() -> None:
    """FinalizationConfig never feeds create_transcriber; recognition does."""
    recognition = recognize_for("faster-whisper", {})
    transcriber = create_transcriber(recognition, "cpu", "de-DE")
    assert isinstance(transcriber, FasterWhisperTranscriber)
    with pytest.raises(ValueError, match="faster_whisper_compute_type"):
        recognize_for("faster-whisper", {"faster_whisper_compute_type": "float8"})


def test_invalid_faster_whisper_setting_does_not_break_parakeet() -> None:
    """Only the selected backend's validation runs."""
    config = recognize_for("parakeet", {"faster_whisper_compute_type": "float8"})
    assert config.asr_backend == "parakeet"


def test_nemotron_lookahead_accepts_an_explicit_integer() -> None:
    """Nemotron validates explicit lookaheads through the loaded processor, not the parser."""
    config = recognize_for("nemotron", {}, {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "-3"})
    assert config.nemotron_num_lookahead_tokens == -3
