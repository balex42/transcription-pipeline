"""Stage-specific configuration: per-stage environment parsing and validation."""

from dataclasses import replace
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
    FinalizationConfig,
    PreparationConfig,
    RecognitionConfig,
)
from speech_transcriber.transcription.factory import create_transcriber
from speech_transcriber.transcription.faster_whisper import FasterWhisperTranscriber


def prepare(overrides: dict[str, object], env: dict[str, str] | None = None) -> PreparationConfig:
    return PreparationConfig.from_environment(Path("in.wav"), Path("out"), overrides, env or {})


def recognize(
    overrides: dict[str, object], env: dict[str, str] | None = None
) -> RecognitionConfig:
    return RecognitionConfig.from_environment(Path("prepared"), Path("out"), overrides, env or {})


def finalize(
    overrides: dict[str, object], env: dict[str, str] | None = None
) -> FinalizationConfig:
    return FinalizationConfig.from_environment(Path("out"), overrides, env or {})


# --- Recognition configuration -------------------------------------------------


def test_recognition_default_backend_maps_to_the_default_model() -> None:
    config = recognize({})
    assert (config.asr_backend, config.resolved_asr_model) == ("parakeet", DEFAULT_PARAKEET_MODEL)
    assert (config.parakeet_segment_duration, config.parakeet_segment_overlap) == (180.0, 15.0)


def test_environment_backend_and_default_model_mapping() -> None:
    assert recognize({}, {"ASR_BACKEND": "nemotron"}).resolved_asr_model == DEFAULT_NEMOTRON_MODEL
    assert recognize({}, {"ASR_BACKEND": "voxtral"}).resolved_asr_model == DEFAULT_VOXTRAL_MODEL
    assert (
        recognize({}, {"ASR_BACKEND": "faster-whisper"}).resolved_asr_model
        == DEFAULT_FASTER_WHISPER_MODEL
    )
    assert recognize({}, {"ASR_BACKEND": "canary"}).resolved_asr_model == DEFAULT_CANARY_MODEL
    assert recognize({}, {"ASR_BACKEND": "primeline"}).resolved_asr_model == (
        DEFAULT_PRIMELINE_MODEL
    )


def test_primeline_explicit_asr_model_override_wins() -> None:
    config = recognize({"asr_backend": "primeline"}, {"ASR_MODEL": "/models/primeline"})
    assert config.resolved_asr_model == "/models/primeline"


def test_faster_whisper_uses_float16_compute_type_by_default() -> None:
    assert (
        recognize({"asr_backend": "faster-whisper"}).faster_whisper_compute_type
        == DEFAULT_FASTER_WHISPER_COMPUTE_TYPE
    )


def test_faster_whisper_compute_type_allows_an_override() -> None:
    config = recognize(
        {"asr_backend": "faster-whisper"},
        {"FASTER_WHISPER_COMPUTE_TYPE": "int8_float16"},
    )
    assert config.faster_whisper_compute_type == "int8_float16"


@pytest.mark.parametrize("compute_type", ["float8", "int4"])
def test_faster_whisper_rejects_unsupported_compute_type(compute_type: str) -> None:
    with pytest.raises(ValueError, match="faster_whisper_compute_type"):
        recognize(
            {"asr_backend": "faster-whisper", "faster_whisper_compute_type": compute_type},
        )


def test_canary_uses_ten_second_chunks_by_default() -> None:
    assert (
        recognize({"asr_backend": "canary"}).canary_chunk_duration_seconds
        == DEFAULT_CANARY_CHUNK_DURATION_SECONDS
    )


def test_canary_chunk_duration_allows_an_override() -> None:
    assert recognize({}, {"CANARY_CHUNK_DURATION": "20"}).canary_chunk_duration_seconds == 20.0
    assert recognize({"canary_chunk_duration_seconds": 15.5}).canary_chunk_duration_seconds == 15.5


@pytest.mark.parametrize("duration", [0, -5])
def test_canary_rejects_nonpositive_chunk_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="canary_chunk_duration_seconds"):
        recognize({"asr_backend": "canary", "canary_chunk_duration_seconds": duration})


def test_nemotron_uses_highest_accuracy_lookahead_by_default() -> None:
    assert (
        recognize({}).nemotron_num_lookahead_tokens == DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS
    )


def test_nemotron_lookahead_allows_a_low_latency_override() -> None:
    assert recognize({}, {"NEMOTRON_NUM_LOOKAHEAD_TOKENS": "0"}).nemotron_num_lookahead_tokens == 0


def test_voxtral_uses_highest_accuracy_delay_by_default() -> None:
    assert recognize({}).voxtral_delay_ms == DEFAULT_VOXTRAL_DELAY_MS


def test_voxtral_delay_allows_a_lower_latency_override() -> None:
    assert recognize({}, {"VOXTRAL_DELAY_MS": "480"}).voxtral_delay_ms == 480


def test_voxtral_uses_calibrated_timestamp_offset_by_default() -> None:
    assert (
        recognize({}).voxtral_timestamp_offset_tokens == DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS
    )


def test_voxtral_timestamp_offset_allows_an_override() -> None:
    assert (
        recognize({}, {"VOXTRAL_TIMESTAMP_OFFSET_TOKENS": "6"}).voxtral_timestamp_offset_tokens
        == 6
    )


def test_voxtral_rejects_delay_outside_supported_range() -> None:
    with pytest.raises(ValueError, match="Voxtral delay must be between"):
        recognize({}, {"VOXTRAL_DELAY_MS": "2401"})


def test_voxtral_rejects_delay_that_is_not_a_multiple_of_80ms() -> None:
    with pytest.raises(ValueError, match="multiple of 80ms"):
        recognize({}, {"VOXTRAL_DELAY_MS": "500"})


@pytest.mark.parametrize("offset", ["-1", "31"])
def test_voxtral_rejects_timestamp_offset_outside_delay_horizon(offset: str) -> None:
    with pytest.raises(ValueError, match="timestamp offset"):
        recognize({}, {"VOXTRAL_TIMESTAMP_OFFSET_TOKENS": offset})


def test_cli_backend_and_model_override_environment() -> None:
    config = recognize(
        {"asr_backend": "qwen", "asr_model": "/models/qwen"},
        {"ASR_BACKEND": "parakeet", "ASR_MODEL": "/models/parakeet"},
    )
    assert (config.asr_backend, config.resolved_asr_model) == ("qwen", "/models/qwen")


def test_qwen_uses_asr_and_forced_aligner_defaults() -> None:
    config = recognize({"asr_backend": "qwen"})
    assert config.resolved_asr_model == DEFAULT_QWEN_MODEL
    assert config.qwen_aligner_model == DEFAULT_QWEN_ALIGNER_MODEL
    assert (config.qwen_segment_duration, config.qwen_segment_overlap) == (240.0, 15.0)


def test_environment_segment_settings_override_qwen_defaults() -> None:
    config = recognize(
        {"asr_backend": "qwen"},
        {"QWEN_SEGMENT_DURATION": "120", "QWEN_SEGMENT_OVERLAP": "20"},
    )
    assert (config.qwen_segment_duration, config.qwen_segment_overlap) == (120.0, 20.0)


def test_qwen_rejects_segments_above_forced_alignment_limit() -> None:
    with pytest.raises(ValueError, match="forced-aligner limit"):
        recognize({"asr_backend": "qwen", "qwen_segment_duration": 301})


# --- Stage environment isolation ------------------------------------------------


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
    ],
)
def test_prepare_ignores_recognition_environment_garbage(garbage: dict[str, str]) -> None:
    config = prepare({"language": "fr-FR"}, garbage)
    assert (config.language, config.pyannote_model) == ("fr-FR", config.pyannote_model)
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
    ],
)
def test_finalize_ignores_recognition_and_diarization_garbage(garbage: dict[str, str]) -> None:
    config = finalize({}, garbage)
    assert config.alignment_tolerance == 0.25
    assert config.turn_gap_seconds == 1.0
    assert config.log_level == "INFO"


@pytest.mark.parametrize(
    "garbage",
    [
        {"NUM_SPEAKERS": "two"},
        {"MIN_SPEAKERS": "lots"},
        {"MAX_SPEAKERS": "many"},
        {"PYANNOTE_MODEL": "anything"},
    ],
)
def test_recognize_ignores_diarization_environment_garbage(garbage: dict[str, str]) -> None:
    config = recognize({}, garbage)
    assert config.asr_backend == "parakeet"
    assert (config.parakeet_segment_duration, config.parakeet_segment_overlap) == (180.0, 15.0)


def test_recognize_never_parses_a_language_environment_variable() -> None:
    """LANGUAGE must not override or inject recognition language; the artifact owns it."""
    assert recognize({}, {"LANGUAGE": "fr-FR"}).language is None


# --- Recognition language inheritance -------------------------------------------


def test_recognition_inherits_french_prepared_language() -> None:
    assert recognize({}).language_for("fr-FR") == "fr-FR"


def test_recognition_inherits_english_prepared_language() -> None:
    assert recognize({}).language_for("en-US") == "en-US"


def test_explicit_recognition_language_overrides_the_prepared_value() -> None:
    assert recognize({"language": "de-DE"}).language_for("fr-FR") == "de-DE"


def test_recognition_has_no_generic_language_default_to_inherit() -> None:
    """A missing prepared language must not be silently replaced by de-DE."""
    assert recognize({}).language_for(None) is None


def test_recognition_language_is_resolved_from_the_prepared_artifact_in_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a fr-FR prepared artifact reaches the backend adapter as fr-FR."""
    from speech_transcriber import cli
    from speech_transcriber.models import (
        ASRWord,
        AudioMetadata,
        DiarizationSegment,
        NormalizedAudio,
    )
    from speech_transcriber.prepared import PreparedRecording, sha256_file, write_prepared_recording

    source = tmp_path / "normalized.wav"
    source.write_bytes(b"normalized audio")
    prepared_directory = tmp_path / "prepared"
    write_prepared_recording(
        PreparedRecording(
            NormalizedAudio(source, AudioMetadata("meeting.wav", 2.0)),
            [DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
            tmp_path,
            normalized_audio_sha256=sha256_file(source),
            diarization_model="pyannote/test",
            language="fr-FR",
        ),
        prepared_directory,
    )

    seen: dict[str, object] = {}

    def fake_transcriber(config: object, device: str) -> object:
        seen["language"] = config.language  # type: ignore[attr-defined]
        seen["device"] = device

        class FakeTranscriber:
            device = "cpu"
            dtype_name = "float32"
            model_reference = "fake/asr"
            backend_metrics: dict[str, float] = {}
            backend_models: dict[str, str] = {}
            backend_configuration: dict[str, object] = {}

            def load(self) -> None: ...

            def transcribe(self, _: object) -> list[ASRWord]:
                return [ASRWord("bonjour", end=0.5, start=0.0)]

            def release(self) -> None: ...

        return FakeTranscriber()

    import speech_transcriber.transcription.factory as factory_module

    monkeypatch.setattr(factory_module, "create_transcriber", fake_transcriber)

    assert (
        cli.main(
            [
                "recognize",
                "--prepared",
                str(prepared_directory),
                "--backend",
                "parakeet",
                "--output",
                str(tmp_path / "asr"),
            ]
        )
        == 0
    )
    assert seen["language"] == "fr-FR"
    assert seen["device"] == "cpu"


# --- Preparation configuration --------------------------------------------------


def test_prepare_parses_speaker_settings_and_defaults() -> None:
    config = prepare({})
    assert (config.device, config.pyannote_model, config.language) == (
        "auto",
        config.pyannote_model,
        "de-DE",
    )
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


# --- Finalization configuration -------------------------------------------------


def test_finalize_defaults_and_overrides() -> None:
    assert (finalize({}).alignment_tolerance, finalize({}).turn_gap_seconds) == (0.25, 1.0)
    config = finalize({"alignment_tolerance": 0.5, "turn_gap_seconds": 2.0})
    assert (config.alignment_tolerance, config.turn_gap_seconds) == (0.5, 2.0)


def test_finalize_validates_against_backends_for_the_selected_transcriber() -> None:
    """FinalizationConfig never feeds create_transcriber; recognition does."""
    recognition = recognize({"asr_backend": "faster-whisper", "language": "de-DE"})
    assert isinstance(create_transcriber(recognition, "cpu"), FasterWhisperTranscriber)
    with pytest.raises(ValueError, match="faster_whisper_compute_type"):
        recognize({"faster_whisper_compute_type": "float8"})


def test_recognition_config_rejects_empty_language_override() -> None:
    with pytest.raises(ValueError, match="language must not be empty"):
        recognize({"language": ""})


def test_replace_preserves_frozen_recognition_config() -> None:
    config = recognize({"language": "de-DE"})
    assert replace(config, language="fr-FR").language == "fr-FR"
    assert config.language == "de-DE"