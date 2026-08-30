"""Unit tests for the NeMo-based Parakeet backend without loading the real model."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from speech_transcriber.config import DEFAULT_PARAKEET_MODEL, PARAKEET_MODEL_FILE
from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.models import AudioMetadata, NormalizedAudio
from speech_transcriber.transcription.parakeet import (
    ParakeetTranscriber,
    flatten_parakeet_words,
    resolve_parakeet_model_path,
)


class FakeHypothesis:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.timestamp: dict[str, object] = {"word": records}


class FakeNeMoModel:
    """Restored-model stand-in capturing segment calls and emitting one word."""

    def __init__(self, records: list[dict[str, object]] | None = None) -> None:
        self.records = (
            records
            if records is not None
            else [{"word": "Wort", "start": 0.0, "end": 0.5}]
        )
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio_arrays: list, **kwargs: object) -> list[object]:
        self.calls.append({"samples": len(audio_arrays[0]), **kwargs})
        return [FakeHypothesis(self.records)]


def make_audio(tmp_path: Path, seconds: float = 3.0) -> NormalizedAudio:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(int(16_000 * seconds), dtype="<i2").tobytes())
    return NormalizedAudio(path, AudioMetadata(path.name, seconds))


def local_checkpoint(tmp_path: Path) -> Path:
    artifact = tmp_path / PARAKEET_MODEL_FILE
    artifact.write_bytes(b"trusted model")
    return artifact


def patch_restore(monkeypatch: pytest.MonkeyPatch, model: object) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def restore(path: str, device: str) -> object:
        calls.append((path, device))
        return model

    import speech_transcriber.transcription.parakeet as module

    monkeypatch.setattr(module, "_restore_parakeet_model", restore)
    return calls


# --- expected checkpoint filename ---


def test_expected_checkpoint_filename_is_exact() -> None:
    assert PARAKEET_MODEL_FILE == "parakeet-tdt-0.6b-v3.nemo"


# --- offline resolution ---


def test_explicit_local_nemo_file_is_used_directly(tmp_path: Path) -> None:
    artifact = tmp_path / PARAKEET_MODEL_FILE
    artifact.write_bytes(b"trusted model")

    assert resolve_parakeet_model_path(str(artifact)) == str(artifact)


def test_explicit_local_directory_locates_the_nemo_checkpoint(tmp_path: Path) -> None:
    (tmp_path / PARAKEET_MODEL_FILE).write_bytes(b"trusted model")

    assert resolve_parakeet_model_path(str(tmp_path)) == str(tmp_path / PARAKEET_MODEL_FILE)


def test_explicit_local_directory_without_the_checkpoint_fails_clearly(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelLoadError, match=f"'{PARAKEET_MODEL_FILE}' is missing"):
        resolve_parakeet_model_path(str(tmp_path))


def test_repository_id_resolves_through_local_hf_cache_refs_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    repo = cache / "hub" / "models--nvidia--parakeet-tdt-0.6b-v3"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / PARAKEET_MODEL_FILE).write_bytes(b"trusted model")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(cache))

    assert (
        resolve_parakeet_model_path(DEFAULT_PARAKEET_MODEL)
        == str(snapshot / PARAKEET_MODEL_FILE)
    )


def test_repository_id_uses_single_snapshot_without_refs_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "models--nvidia--parakeet-tdt-0.6b-v3" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / PARAKEET_MODEL_FILE).write_bytes(b"trusted model")
    monkeypatch.setenv("HF_HOME", str(cache))

    assert resolve_parakeet_model_path(DEFAULT_PARAKEET_MODEL) == str(
        snapshot / PARAKEET_MODEL_FILE
    )


def test_repository_id_fails_without_online_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_parakeet_model_path(DEFAULT_PARAKEET_MODEL)


def test_ambiguous_snapshots_fail_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = (
        tmp_path / "cache" / "hub" / "models--nvidia--parakeet-tdt-0.6b-v3" / "snapshots"
    )
    (snapshots / "old123").mkdir(parents=True)
    (snapshots / "new456").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))

    with pytest.raises(ModelLoadError, match="refusing to guess"):
        resolve_parakeet_model_path(DEFAULT_PARAKEET_MODEL)


# --- lazy NeMo restoration and provenance ---


def test_load_restores_the_local_checkpoint_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / PARAKEET_MODEL_FILE
    artifact.write_bytes(b"trusted model")
    model = FakeNeMoModel()
    calls: list[tuple[str, str]] = []

    def restore(path: str, device: str) -> object:
        calls.append((path, device))
        return model

    import speech_transcriber.transcription.parakeet as module

    monkeypatch.setattr(module, "_restore_parakeet_model", restore)
    instance = ParakeetTranscriber(str(artifact), "cuda")

    instance.load()

    assert calls == [(str(artifact), "cuda")]
    assert instance.backend_models["model_file"] == PARAKEET_MODEL_FILE
    assert "model_snapshot" not in instance.backend_models
    assert instance.runtime_provenance.name == "nemo"
    assert "torch" in instance.runtime_provenance.components
    assert "cuda" in instance.runtime_provenance.components
    assert instance.backend_configuration["checkpoint_file"] == PARAKEET_MODEL_FILE
    assert instance.dtype_name == "checkpoint-default"


def test_load_records_cached_snapshot_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    repo = cache / "hub" / "models--nvidia--parakeet-tdt-0.6b-v3"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / PARAKEET_MODEL_FILE).write_bytes(b"trusted model")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(cache))
    import speech_transcriber.transcription.parakeet as module

    monkeypatch.setattr(module, "_restore_parakeet_model", lambda path, device: FakeNeMoModel())
    instance = ParakeetTranscriber(DEFAULT_PARAKEET_MODEL, "cuda")

    instance.load()

    assert instance.backend_models["model_snapshot"] == "abc123"
    assert instance.backend_models["model_file"] == PARAKEET_MODEL_FILE


def test_load_wraps_nemo_failures_as_model_load_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / PARAKEET_MODEL_FILE
    artifact.write_bytes(b"trusted model")
    import speech_transcriber.transcription.parakeet as module

    def restore(path: str, device: str) -> object:
        raise RuntimeError("nemo exploded")

    monkeypatch.setattr(module, "_restore_parakeet_model", restore)
    instance = ParakeetTranscriber(str(artifact), "cpu")

    with pytest.raises(ModelLoadError, match="could not load Parakeet model"):
        instance.load()


# --- native word timestamps ---


def test_flat_word_timestamps_become_canonical_words() -> None:
    words = flatten_parakeet_words(
        [
            FakeHypothesis(
                [
                    {"word": "Guten", "start": 0.0, "end": 0.4},
                    {"word": "Morgen!", "start": 0.4, "end": 1.0},
                ]
            )
        ]
    )

    assert [(word.text, word.start, word.end, word.confidence) for word in words] == [
        ("Guten", 0.0, 0.4, None),
        ("Morgen!", 0.4, 1.0, None),
    ]


def test_numeric_confidence_is_preserved_when_exposed() -> None:
    words = flatten_parakeet_words(
        [FakeHypothesis([{"word": "Hallo", "start": 0.0, "end": 0.5, "confidence": 0.97}])]
    )

    assert words[0].confidence == 0.97


def test_multiple_hypotheses_are_rejected() -> None:
    with pytest.raises(ASROutputError, match="2 hypotheses for one segment"):
        flatten_parakeet_words([FakeHypothesis([]), FakeHypothesis([])])


def test_missing_timestamp_metadata_fails_clearly() -> None:
    with pytest.raises(ASROutputError, match="missing timestamp metadata"):
        flatten_parakeet_words([object()])


def test_missing_word_timestamps_fail_clearly() -> None:
    fake = FakeHypothesis([])
    fake.timestamp = {"segment": []}
    with pytest.raises(ASROutputError, match="missing word timestamps"):
        flatten_parakeet_words([fake])


def test_empty_word_timestamp_list_is_valid_silence() -> None:
    assert flatten_parakeet_words([FakeHypothesis([])]) == []


def test_empty_word_text_fails_clearly() -> None:
    with pytest.raises(ASROutputError, match="missing text"):
        flatten_parakeet_words(
            [FakeHypothesis([{"word": "  ", "start": 0.0, "end": 0.5}])]
        )


def test_non_numeric_bounds_fail_clearly() -> None:
    with pytest.raises(ASROutputError, match="missing numeric start/end"):
        flatten_parakeet_words(
            [FakeHypothesis([{"word": "Hallo", "start": None, "end": 0.5}])]  # type: ignore[list-item]
        )


def test_end_before_start_fails_clearly() -> None:
    with pytest.raises(ASROutputError, match="invalid"):
        flatten_parakeet_words([FakeHypothesis([{"word": "Hallo", "start": 0.5, "end": 0.4}])])


def test_non_monotonic_timestamps_fail_clearly() -> None:
    with pytest.raises(ASROutputError, match="reorder"):
        flatten_parakeet_words(
            [
                FakeHypothesis(
                    [
                        {"word": "zweite", "start": 0.8, "end": 1.0},
                        {"word": "erste", "start": 0.2, "end": 0.5},
                    ]
                )
            ]
        )


# --- long-form segmentation strategy stays 180/15 ---


def test_defaults_keep_the_established_segmentation() -> None:
    transcriber = ParakeetTranscriber(DEFAULT_PARAKEET_MODEL, "cpu")

    assert transcriber.backend_configuration["segment_duration_seconds"] == 180.0
    assert transcriber.backend_configuration["segment_overlap_seconds"] == 15.0
    assert transcriber._segmenter.duration_seconds == 180.0
    assert transcriber._segmenter.overlap_seconds == 15.0


def test_segment_settings_allow_existing_overrides() -> None:
    transcriber = ParakeetTranscriber(DEFAULT_PARAKEET_MODEL, "cpu", 120.0, 30.0)

    assert transcriber.backend_configuration["segment_duration_seconds"] == 120.0
    assert transcriber.backend_configuration["segment_overlap_seconds"] == 30.0


# --- segment inference: one load, one numpy call per segment, global rebasing ---


def test_transcribe_loads_once_and_transcribes_each_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeNeMoModel()
    import speech_transcriber.transcription.parakeet as module

    monkeypatch.setattr(module, "_restore_parakeet_model", lambda path, device: model)
    transcriber = ParakeetTranscriber(str(local_checkpoint(tmp_path)), "cpu", 2.0, 1.0)

    words = transcriber.transcribe(make_audio(tmp_path))

    # 3 s audio, 2 s segments, 1 s overlap -> exactly two segment calls.
    assert len(model.calls) == 2
    assert model.calls[0]["samples"] == 32_000
    assert model.calls[1]["samples"] == 32_000
    assert all(call["batch_size"] == 1 for call in model.calls)
    assert all(call["return_hypotheses"] is True for call in model.calls)
    assert all(call["timestamps"] is True for call in model.calls)
    assert transcriber.backend_metrics["segments_processed"] == 2.0
    # The same model object served every segment (loaded once).
    assert transcriber._model is model
    # Segment-local word rebased to each segment's global offset.
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Wort", 0.0, 0.5),
        ("Wort", 1.0, 1.5),
    ]


def test_transcribe_wraps_nemo_failures_as_output_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingModel:
        def transcribe(self, audio_arrays: list, **kwargs: object) -> list[object]:
            raise RuntimeError("nemo exploded")

    import speech_transcriber.transcription.parakeet as module

    monkeypatch.setattr(module, "_restore_parakeet_model", lambda path, device: ExplodingModel())
    transcriber = ParakeetTranscriber(str(local_checkpoint(tmp_path)), "cpu", 2.0, 1.0)

    with pytest.raises(ASROutputError, match="Parakeet recognition failed for segment"):
        transcriber.transcribe(make_audio(tmp_path))


# --- no Transformers adapter usage remains ---


def test_parakeet_module_no_longer_uses_transformers() -> None:
    source = (
        Path(__file__).parents[2] / "src/speech_transcriber/transcription/parakeet.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "AutoProcessor",
        "AutoModelForTDT",
        "from transformers",
        "import transformers",
        "processor.decode",
        "forced_alignment",
    ):
        assert forbidden not in source, forbidden
