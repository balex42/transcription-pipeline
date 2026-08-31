"""Unit tests for the NeMo-based Primeline backend without loading the real model."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.models import AudioMetadata, NormalizedAudio
from speech_transcriber.transcription.primeline import (
    PrimelineTranscriber,
    flatten_primeline_words,
    resolve_primeline_model_path,
)


class Hypothesis:
    def __init__(self, records: list[dict[str, object]] | None = None) -> None:
        self.timestamp: dict[str, object] | None = {"word": records or []}
        if records is not None:
            self.timestamp = {"word": records}


def hypothesis(records: list[dict[str, object]]) -> object:
    instance = object.__new__(Hypothesis)
    instance.timestamp = {"word": records}
    return instance


def audio(tmp_path: Path, seconds: float = 1.0) -> NormalizedAudio:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * round(16_000 * seconds))
    return NormalizedAudio(path, AudioMetadata(path.name, seconds))


def fake_repository_cache(tmp_path: Path) -> Path:
    """Build a prefetched single-revision HF cache for the Primeline model."""
    snapshot = (
        tmp_path / "cache" / "hub" / "models--primeline--parakeet-primeline"
        / "snapshots" / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "2_95_WER.nemo").write_bytes(b"trusted model")
    refs = tmp_path / "cache" / "hub" / "models--primeline--parakeet-primeline" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc123\n", encoding="utf-8")
    return snapshot


def test_repository_id_resolves_through_local_hf_cache_refs_main(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "cache"
    repo = cache / "hub" / "models--primeline--parakeet-primeline"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "2_95_WER.nemo").write_bytes(b"trusted model")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(cache))  # type: ignore[attr-defined]

    assert (
        resolve_primeline_model_path("primeline/parakeet-primeline")
        == str(snapshot / "2_95_WER.nemo")
    )


def test_repository_id_uses_single_snapshot_without_refs_main(
    tmp_path: Path, monkeypatch: object
) -> None:
    snapshot = (
        tmp_path / "cache" / "hub" / "models--primeline--parakeet-primeline"
        / "snapshots" / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "2_95_WER.nemo").write_bytes(b"trusted model")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))  # type: ignore[attr-defined]

    assert (
        resolve_primeline_model_path("primeline/parakeet-primeline")
        == str(snapshot / "2_95_WER.nemo")
    )


def test_repository_id_fails_without_online_fallback(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))  # type: ignore[attr-defined]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # type: ignore[attr-defined]

    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_primeline_model_path("primeline/parakeet-primeline")


def test_ambiguous_snapshots_fail_deterministically(tmp_path: Path, monkeypatch: object) -> None:
    snapshots = (
        tmp_path / "cache" / "hub" / "models--primeline--parakeet-primeline" / "snapshots"
    )
    (snapshots / "old123").mkdir(parents=True)
    (snapshots / "new456").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))  # type: ignore[attr-defined]

    with pytest.raises(ModelLoadError, match="refusing to guess"):
        resolve_primeline_model_path("primeline/parakeet-primeline")


def test_explicit_local_nemo_file_is_used_directly(tmp_path: Path) -> None:
    artifact = tmp_path / "2_95_WER.nemo"
    artifact.write_bytes(b"trusted model")

    assert resolve_primeline_model_path(str(artifact)) == str(artifact)


def test_explicit_local_directory_locates_the_nemo_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "2_95_WER.nemo").write_bytes(b"trusted model")

    assert resolve_primeline_model_path(str(tmp_path)) == str(tmp_path / "2_95_WER.nemo")


def test_explicit_local_directory_without_the_checkpoint_fails_clearly(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelLoadError, match="'2_95_WER.nemo' is missing"):
        resolve_primeline_model_path(str(tmp_path))


def test_flat_word_timestamps_become_canonical_words() -> None:
    words = flatten_primeline_words(
        [
            Hypothesis(
                [
                    {"word": "Guten", "start": 0.0, "end": 0.4},
                    {"word": "Morgen!", "start": 0.4, "end": 1.0},
                ]
            )
        ],
        duration_seconds=1.0,
    )

    assert [(word.text, word.start, word.end, word.confidence) for word in words] == [
        ("Guten", 0.0, 0.4, None),
        ("Morgen!", 0.4, 1.0, None),
    ]


def test_numeric_confidence_is_preserved_when_exposed() -> None:
    words = flatten_primeline_words(
        [Hypothesis([{"word": "Hallo", "start": 0.0, "end": 0.5, "confidence": 0.97}])]
    )

    assert words[0].confidence == 0.97


def test_multiple_hypotheses_are_rejected() -> None:
    with pytest.raises(ASROutputError, match="2 hypotheses for one segment"):
        flatten_primeline_words([Hypothesis([]), Hypothesis([])])


def test_missing_timestamp_metadata_fails_clearly() -> None:
    with pytest.raises(ASROutputError, match="missing timestamp metadata"):
        flatten_primeline_words([object()])


def test_missing_word_timestamps_fail_clearly() -> None:
    hypothesis = Hypothesis([])
    hypothesis.timestamp = {"segment": []}
    with pytest.raises(ASROutputError, match="missing word timestamps"):
        flatten_primeline_words([hypothesis])


def test_empty_word_timestamp_list_is_valid_silence() -> None:
    assert flatten_primeline_words([Hypothesis([])]) == []


def test_empty_word_text_fails_clearly() -> None:
    with pytest.raises(ASROutputError, match="missing text"):
        flatten_primeline_words([Hypothesis([{"word": "  ", "start": 0.0, "end": 0.5}])])


def test_non_numeric_bounds_fail_clearly() -> None:
    with pytest.raises(ASROutputError, match="missing numeric start/end"):
        flatten_primeline_words([Hypothesis([{"word": "Hallo", "start": None, "end": 0.5}])])  # type: ignore[list-item]


def test_end_before_start_fails_clearly() -> None:
    with pytest.raises(ASROutputError, match="invalid"):
        flatten_primeline_words([Hypothesis([{"word": "Hallo", "start": 0.5, "end": 0.4}])])


def test_negative_start_fails_clearly() -> None:
    with pytest.raises(ASROutputError, match="invalid"):
        flatten_primeline_words([Hypothesis([{"word": "Hallo", "start": -0.1, "end": 0.5}])])


def test_non_monotonic_timestamps_fail_clearly() -> None:
    with pytest.raises(ASROutputError, match="reorder"):
        flatten_primeline_words(
            [
                Hypothesis(
                    [
                        {"word": "zweite", "start": 0.8, "end": 1.0},
                        {"word": "erste", "start": 0.2, "end": 0.5},
                    ]
                )
            ]
        )


def test_timestamps_exceeding_the_recording_duration_fail() -> None:
    with pytest.raises(ASROutputError, match="exceeds recording duration"):
        flatten_primeline_words(
            [Hypothesis([{"word": "Hallo", "start": 0.0, "end": 2.5}])],
            duration_seconds=1.0,
        )


def test_load_restores_the_local_checkpoint_and_records_nemo_provenance(
    tmp_path: Path, monkeypatch: object
) -> None:
    artifact = tmp_path / "2_95_WER.nemo"
    artifact.write_bytes(b"trusted model")
    calls: list[tuple[str, str]] = []

    class RestoredModel:
        def to(self, _: str) -> RestoredModel:
            return self

        def eval(self) -> RestoredModel:
            return self

    def restore(model_path: str, device: str) -> RestoredModel:
        calls.append((model_path, device))
        return RestoredModel()

    import speech_transcriber.transcription.primeline as module

    monkeypatch.setattr(module, "_restore_primeline_model", restore)  # type: ignore[attr-defined]
    instance = PrimelineTranscriber(str(artifact), "cuda")

    instance.load()

    assert calls == [(str(artifact), "cuda")]
    assert instance.backend_models["model_file"] == "2_95_WER.nemo"
    assert "model_snapshot" not in instance.backend_models
    assert instance.runtime_provenance.name == "nemo"
    assert "torch" in instance.runtime_provenance.components
    assert "cuda" in instance.runtime_provenance.components
    assert instance.backend_configuration["checkpoint_file"] == "2_95_WER.nemo"
    assert instance.dtype_name == "checkpoint-default"


def test_transcribe_uses_the_shared_segmented_nemo_path_and_validates_output(
    tmp_path: Path, monkeypatch: object
) -> None:
    class RestoredModel:
        def to(self, _: str) -> RestoredModel:
            return self

        def eval(self) -> RestoredModel:
            return self

    calls: list[dict[str, object]] = []

    def restore(model_path: str, device: str) -> RestoredModel:
        instance = RestoredModel()

        def transcribe(samples: list[np.ndarray], **kwargs: object) -> list[Hypothesis]:
            calls.append({"samples": len(samples[0]), **kwargs})
            return [
                Hypothesis(
                    [
                        {"word": "Hallo", "start": 0.0, "end": 0.4},
                        {"word": "Welt!", "start": 0.4, "end": 0.9},
                    ]
                )
            ]

        instance.transcribe = transcribe  # type: ignore[method-assign]
        return instance

    import speech_transcriber.transcription.primeline as module

    monkeypatch.setattr(module, "_restore_primeline_model", restore)  # type: ignore[attr-defined]
    fake_repository_cache(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))  # type: ignore[attr-defined]
    transcriber = PrimelineTranscriber("primeline/parakeet-primeline", "cuda")

    words = transcriber.transcribe(audio(tmp_path))

    assert [(w.text, w.start, w.end) for w in words] == [
        ("Hallo", 0.0, 0.4),
        ("Welt!", 0.4, 0.9),
    ]
    assert calls[0]["samples"] == 16_000
    assert calls[0]["batch_size"] == 1
    assert calls[0]["return_hypotheses"] is True
    assert calls[0]["timestamps"] is True
    assert transcriber.backend_metrics["segments_processed"] == 1.0


def test_transcribe_rebases_overlapping_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Model:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def transcribe(self, samples: list[np.ndarray], **kwargs: object) -> list[Hypothesis]:
            self.calls.append(len(samples[0]))
            return [Hypothesis([{"word": "Hallo", "start": 0.0, "end": 0.5}])]

    model = Model()
    artifact = tmp_path / "2_95_WER.nemo"
    artifact.write_bytes(b"trusted model")
    import speech_transcriber.transcription.primeline as module

    monkeypatch.setattr(module, "_restore_primeline_model", lambda path, device: model)
    transcriber = PrimelineTranscriber(str(artifact), "cpu", 2.0, 1.0)

    words = transcriber.transcribe(audio(tmp_path, seconds=3.0))

    assert model.calls == [32_000, 32_000]
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Hallo", 0.0, 0.5),
        ("Hallo", 1.0, 1.5),
    ]
    assert transcriber.backend_metrics["segments_processed"] == 2.0


def test_transcribe_wraps_nemo_failures_as_output_errors(
    tmp_path: Path, monkeypatch: object
) -> None:
    class RestoredModel:
        def to(self, _: str) -> RestoredModel:
            return self

        def eval(self) -> RestoredModel:
            return self

    def restore(model_path: str, device: str) -> RestoredModel:
        instance = RestoredModel()

        def transcribe(audio: list[str], **kwargs: object) -> list[Hypothesis]:
            raise RuntimeError("nemo exploded")

        instance.transcribe = transcribe  # type: ignore[method-assign]
        return instance

    import speech_transcriber.transcription.primeline as module

    monkeypatch.setattr(module, "_restore_primeline_model", restore)  # type: ignore[attr-defined]
    fake_repository_cache(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))  # type: ignore[attr-defined]
    transcriber = PrimelineTranscriber("primeline/parakeet-primeline", "cpu")

    with pytest.raises(ASROutputError, match="Primeline recognition failed"):
        transcriber.transcribe(audio(tmp_path))
