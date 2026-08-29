from __future__ import annotations

from pathlib import Path

import pytest

from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.models import AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.prepared import PreparedRecording, sha256_file
from speech_transcriber.recognition import RecognitionRunner
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.canary import (
    CANARY_MODEL_FILE,
    CanaryTranscriber,
    canary_language,
    flatten_canary_words,
    resolve_canary_model_path,
)


class Hypothesis:
    def __init__(
        self, words: list[dict[str, object]], segments: list[dict[str, object]] | None = None
    ) -> None:
        self.timestamp = {"word": words, "segment": segments or []}


class Model:
    def __init__(self, output: Hypothesis) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio: list[str], **kwargs: object) -> list[Hypothesis]:
        self.calls.append({"audio": audio, **kwargs})
        return [self.output]


def audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "normalized.wav"
    path.write_bytes(b"normalized audio")
    return NormalizedAudio(path, AudioMetadata("meeting.wav", 45.0))


def test_capabilities_declare_native_timestamps_and_no_forced_alignment() -> None:
    transcriber = CanaryTranscriber("nvidia/canary-1b-v2", "cuda", "de-DE")

    assert transcriber.capabilities == TranscriberCapabilities(True, True, True, True)
    assert transcriber.capabilities.streaming is False
    assert transcriber.capabilities.requires_forced_alignment is False
    assert transcriber.dtype_name == "checkpoint-default"


def test_transcribe_uses_german_asr_with_native_word_timestamps(tmp_path: Path) -> None:
    model = Model(
        Hypothesis(
            [
                {"word": "Hallo", "start": 0.0, "end": 0.4, "start_offset": 0, "end_offset": 5},
                {"word": "Welt!", "start": 0.4, "end": 0.9, "start_offset": 5, "end_offset": 11},
            ],
            [{"segment": "Hallo Welt!", "start": 0.0, "end": 0.9}],
        )
    )
    transcriber = CanaryTranscriber("nvidia/canary-1b-v2", "cuda", "de-DE")
    transcriber._model = model

    words = transcriber.transcribe(audio(tmp_path))

    assert [(word.text, word.start, word.end, word.confidence) for word in words] == [
        ("Hallo", 0.0, 0.4, None),
        ("Welt!", 0.4, 0.9, None),
    ]
    assert model.calls == [
        {
            "audio": [str(tmp_path / "normalized.wav")],
            "batch_size": 1,
            "return_hypotheses": True,
            "source_lang": "de",
            "target_lang": "de",
            "timestamps": True,
        }
    ]
    assert transcriber.backend_configuration == {
        "requested_language": "de-DE",
        "source_language": "de",
        "target_language": "de",
        "timestamps": True,
        "batch_size": 1,
        "long_form_mode": "native_dynamic_chunking",
    }
    assert transcriber.backend_metrics == {"word_count": 2.0}


def test_transcribe_preserves_absolute_words_from_native_long_form_merge(tmp_path: Path) -> None:
    transcriber = CanaryTranscriber("nvidia/canary-1b-v2", "cuda", "de")
    transcriber._model = Model(
        Hypothesis(
            [
                {"word": "Anfang", "start": 0.0, "end": 0.5},
                {"word": "Fortsetzung.", "start": 41.1, "end": 42.0},
            ]
        )
    )

    words = transcriber.transcribe(audio(tmp_path))

    assert [(word.text, word.start, word.end) for word in words] == [
        ("Anfang", 0.0, 0.5),
        ("Fortsetzung.", 41.1, 42.0),
    ]


def test_recognition_runner_preserves_prepared_sha_and_canary_metadata(tmp_path: Path) -> None:
    normalized = audio(tmp_path)
    transcriber = CanaryTranscriber("nvidia/canary-1b-v2", "cuda", "de-DE")
    transcriber._model = Model(Hypothesis([{"word": "Hallo", "start": 0.0, "end": 0.5}]))
    prepared = PreparedRecording(
        audio=normalized,
        diarization=[DiarizationSegment("SPEAKER_00", 0.0, 45.0)],
        work_directory=tmp_path,
        normalized_audio_sha256=sha256_file(normalized.path),
        diarization_model="pyannote/test",
        language="de-DE",
        cleanup_enabled=False,
    )

    result = RecognitionRunner().recognize(prepared, transcriber, "canary")

    assert result.metadata.backend == "canary"
    assert result.metadata.model == "nvidia/canary-1b-v2"
    assert result.metadata.normalized_audio_sha256 == prepared.normalized_audio_sha256
    assert result.metadata.runtime.name == "nemo"
    assert result.metadata.backend_configuration["source_language"] == "de"
    assert result.metadata.backend_configuration["target_language"] == "de"


@pytest.mark.parametrize(
    ("language", "expected"),
    [("de-DE", "de"), ("de", "de"), ("DE-de", "de")],
)
def test_canary_language_normalizes_locale(language: str, expected: str) -> None:
    assert canary_language(language) == expected


def test_canary_language_rejects_unsupported_code() -> None:
    with pytest.raises(ValueError, match="does not support language"):
        canary_language("ja-JP")


def test_flatten_canary_words_rejects_invalid_native_timestamp_records() -> None:
    with pytest.raises(ASROutputError, match="missing numeric start/end"):
        flatten_canary_words([Hypothesis([{"word": "Hallo", "start": None, "end": 0.2}])])
    with pytest.raises(ASROutputError, match="ends before"):
        flatten_canary_words([Hypothesis([{"word": "Hallo", "start": 0.3, "end": 0.2}])])


def test_canary_import_is_lazy() -> None:
    import sys

    sys.modules.pop("nemo", None)
    sys.modules.pop("nemo.collections", None)
    assert "nemo" not in sys.modules

    CanaryTranscriber("nvidia/canary-1b-v2", "cuda", "de")

    assert "nemo" not in sys.modules


def test_load_restores_direct_local_nemo_and_records_nemo_provenance(
    tmp_path: Path, monkeypatch: object
) -> None:
    artifact = tmp_path / CANARY_MODEL_FILE
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

    import speech_transcriber.transcription.canary as module

    monkeypatch.setattr(module, "_restore_canary_model", restore)  # type: ignore[attr-defined]
    transcriber = CanaryTranscriber(str(artifact), "cuda", "de")

    transcriber.load()

    assert calls == [(str(artifact), "cuda")]
    assert transcriber.backend_models["model_file"] == str(artifact)
    assert transcriber.runtime_provenance.name == "nemo"
    assert "torch" in transcriber.runtime_provenance.components
    assert "cuda" in transcriber.runtime_provenance.components


def test_resolve_canary_model_path_uses_direct_local_file(tmp_path: Path) -> None:
    artifact = tmp_path / CANARY_MODEL_FILE
    artifact.write_bytes(b"trusted model")

    assert resolve_canary_model_path(str(artifact)) == str(artifact)


def test_resolve_canary_model_path_uses_cached_main_ref(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "cache"
    repo = cache / "hub" / "models--nvidia--canary-1b-v2"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / CANARY_MODEL_FILE).write_bytes(b"trusted model")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(cache))

    assert resolve_canary_model_path("nvidia/canary-1b-v2") == str(snapshot / CANARY_MODEL_FILE)


def test_resolve_canary_model_path_uses_main_ref_with_multiple_snapshots(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "cache"
    repo = cache / "hub" / "models--nvidia--canary-1b-v2"
    old_snapshot = repo / "snapshots" / "old123"
    current_snapshot = repo / "snapshots" / "new456"
    old_snapshot.mkdir(parents=True)
    current_snapshot.mkdir(parents=True)
    (old_snapshot / CANARY_MODEL_FILE).write_bytes(b"old model")
    (current_snapshot / CANARY_MODEL_FILE).write_bytes(b"current model")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("new456\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(cache))

    assert resolve_canary_model_path("nvidia/canary-1b-v2") == str(
        current_snapshot / CANARY_MODEL_FILE
    )


def test_resolve_canary_model_path_uses_single_cached_snapshot(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "models--nvidia--canary-1b-v2" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / CANARY_MODEL_FILE).write_bytes(b"trusted model")
    monkeypatch.setenv("HF_HOME", str(cache))

    assert resolve_canary_model_path("nvidia/canary-1b-v2") == str(snapshot / CANARY_MODEL_FILE)


def test_resolve_canary_model_path_fails_clearly_on_offline_cache_miss(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_canary_model_path("nvidia/canary-1b-v2")


def test_resolve_canary_model_path_rejects_cached_snapshot_without_nemo_artifact(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "cache"
    repo = cache / "hub" / "models--nvidia--canary-1b-v2"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(cache))

    with pytest.raises(ModelLoadError, match="artifact 'canary-1b-v2.nemo' is missing"):
        resolve_canary_model_path("nvidia/canary-1b-v2")
