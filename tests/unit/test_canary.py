from __future__ import annotations

import wave
from pathlib import Path

import pytest

from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.models import ASRWord, AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.prepared import PreparedRecording, sha256_file
from speech_transcriber.recognition import RecognitionRunner
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.canary import (
    CANARY_MODEL_FILE,
    CanaryTranscriber,
    canary_language,
    create_canary_chunks,
    flatten_canary_words,
    frames_per_chunk,
    resolve_canary_model_path,
    validate_canary_words,
    write_canary_chunk,
)

SAMPLE_RATE = 16000


class Hypothesis:
    def __init__(
        self, words: list[dict[str, object]], segments: list[dict[str, object]] | None = None
    ) -> None:
        self.timestamp = {"word": words, "segment": segments or []}


class Model:
    def __init__(self, outputs: list[Hypothesis]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []
        self.transcribed_frames: list[int] = []

    def transcribe(self, audio: list[str], **kwargs: object) -> list[Hypothesis]:
        self.calls.append({"audio": audio, **kwargs})
        with wave.open(audio[0], "rb") as chunk_file:
            self.transcribed_frames.append(chunk_file.getnframes())
        return [self.outputs[len(self.calls) - 1]]


def pcm_audio(tmp_path: Path, seconds: float) -> NormalizedAudio:
    """Create a normalized 16 kHz mono 16-bit PCM WAV fixture of the duration."""
    path = tmp_path / "normalized.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(b"\x00\x00" * int(seconds * SAMPLE_RATE))
    return NormalizedAudio(path, AudioMetadata("meeting.wav", seconds))


def chunk_file_paths(model: Model) -> list[str]:
    return [str(call["audio"][0]) for call in model.calls]


def make_transcriber(tmp_path: Path, chunk_duration: float = 10.0) -> CanaryTranscriber:
    return CanaryTranscriber(
        "nvidia/canary-1b-v2",
        "cuda",
        "de-DE",
        chunk_duration,
        working_directory=tmp_path,
    )


def test_capabilities_declare_native_timestamps_and_no_forced_alignment() -> None:
    transcriber = CanaryTranscriber("nvidia/canary-1b-v2", "cuda", "de-DE", 10.0)

    assert transcriber.capabilities == TranscriberCapabilities(True, True, True, True)
    assert transcriber.capabilities.streaming is False
    assert transcriber.capabilities.requires_forced_alignment is False
    assert transcriber.dtype_name == "checkpoint-default"


def test_transcribe_chunks_and_rebases_native_word_timestamps(tmp_path: Path) -> None:
    model = Model(
        [
            Hypothesis([{"word": "Hallo", "start": 2.0, "end": 2.5}]),
            Hypothesis([{"word": "Welt.", "start": 1.25, "end": 1.8}]),
            Hypothesis([{"word": "Ende", "start": 0.1, "end": 0.55}]),
        ]
    )
    instance = make_transcriber(tmp_path)
    instance._model = model

    words = instance.transcribe(pcm_audio(tmp_path, 25.0))

    assert [(word.text, word.start, word.end) for word in words] == [
        ("Hallo", 2.0, 2.5),
        ("Welt.", 11.25, 11.8),
        ("Ende", 20.1, 20.55),
    ]
    assert all(word.confidence is None for word in words)
    assert len(model.calls) == 3
    for call in model.calls:
        assert call == {
            "audio": [call["audio"][0]],
            "batch_size": 1,
            "return_hypotheses": True,
            "source_lang": "de",
            "target_lang": "de",
            "timestamps": True,
        }
        assert "/canary-chunks." in str(call["audio"][0])
    metrics = dict(instance.backend_metrics)
    del metrics["chunk_transcription_seconds"]
    assert metrics == {
        "word_count": 3.0,
        "chunk_count": 3.0,
        "chunk_duration_seconds": 10.0,
    }
    assert instance.backend_metrics["chunk_transcription_seconds"] >= 0.0
    assert instance.backend_configuration["inference_mode"] == "sequential_non_overlapping_chunks"
    assert instance.backend_configuration["chunk_count"] == 3
    assert instance.backend_configuration["chunk_duration_seconds"] == 10.0


def test_short_recording_still_uses_the_chunk_path(tmp_path: Path) -> None:
    model = Model([Hypothesis([{"word": "Kurz", "start": 0.0, "end": 0.3}])])
    instance = make_transcriber(tmp_path)
    instance._model = model

    words = instance.transcribe(pcm_audio(tmp_path, 5.5))

    assert [(word.text, word.start, word.end, word.confidence) for word in words] == [
        ("Kurz", 0.0, 0.3, None)
    ]
    assert len(model.calls) == 1
    generated = chunk_file_paths(model)
    assert len(generated) == 1
    assert "/canary-chunks." in generated[0]
    assert model.transcribed_frames == [int(5.5 * SAMPLE_RATE)]
    assert instance.backend_metrics["chunk_count"] == 1.0


def test_exact_chunk_duration_yields_one_chunk(tmp_path: Path) -> None:
    model = Model([Hypothesis([{"word": "Hallo", "start": 0.0, "end": 0.4}])])
    instance = make_transcriber(tmp_path)
    instance._model = model

    instance.transcribe(pcm_audio(tmp_path, 10.0))

    assert len(model.calls) == 1
    assert instance.backend_metrics["chunk_count"] == 1.0


def test_long_recording_calls_transcribe_once_per_chunk_and_loads_model_once(
    tmp_path: Path, monkeypatch: object
) -> None:
    import speech_transcriber.transcription.canary as module

    class Model:
        def __init__(self, outputs: list[Hypothesis]) -> None:
            self.outputs = list(outputs)
            self.calls: list[dict[str, object]] = []

        def transcribe(self, audio: list[str], **kwargs: object) -> list[Hypothesis]:
            self.calls.append({"audio": audio, **kwargs})
            return [self.outputs[len(self.calls) - 1]]

    restored_model = Model([Hypothesis([{"word": "w", "start": 0.1, "end": 0.4}])] * 25)
    restore_count = {"count": 0}

    class RestoredModel:
        def __init__(self, delegate: Model) -> None:
            self.delegate = delegate

        def to(self, _: str) -> RestoredModel:
            return self

        def eval(self) -> RestoredModel:
            return self

        def transcribe(self, audio: list[str], **kwargs: object) -> list[Hypothesis]:
            return self.delegate.transcribe(audio, **kwargs)

    wrapper = RestoredModel(restored_model)

    def restore(model_path: str, device: str) -> RestoredModel:
        restore_count["count"] += 1
        return wrapper

    monkeypatch.setattr(
        module, "_restore_canary_model", restore
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(
        module,
        "resolve_canary_model_path",
        lambda model: f"/models/{model}/canary-1b-v2.nemo",
    )

    # The adapter loads lazily inside transcribe() when the model is unset, so
    # a fresh transcriber exercises the real load-once path.
    instance = make_transcriber(tmp_path)
    words = instance.transcribe(pcm_audio(tmp_path, 245.0))

    assert restore_count["count"] == 1
    assert len(restored_model.calls) == 25
    assert instance.backend_metrics["chunk_count"] == 25.0
    assert len(words) == 25
    assert instance._model is wrapper


@pytest.mark.parametrize(
    ("seconds", "expected_starts"),
    [
        (6.0, [0]),
        (10.0, [0]),
        (30.0, [0, 160000, 320000]),
        (25.0, [0, 160000, 320000]),
    ],
)
def test_chunk_plan_shapes(tmp_path: Path, seconds: float, expected_starts: list[int]) -> None:
    path = tmp_path / f"audio-{seconds}.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(b"\x00\x00" * int(seconds * SAMPLE_RATE))

    chunks = [chunk for chunk, _ in create_canary_chunks(path, 10.0)]
    frames = int(seconds * SAMPLE_RATE)

    assert [chunk.start_frame for chunk in chunks] == expected_starts
    assert sum(chunk.frame_count for chunk in chunks) == frames
    final = chunks[-1]
    assert 0 < final.frame_count <= 160000
    assert [chunk.start_frame / SAMPLE_RATE for chunk in chunks] == [
        start / SAMPLE_RATE for start in expected_starts
    ]


def test_partial_final_chunk_shortens_to_remaining_audio(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(b"\x00\x00" * (25 * SAMPLE_RATE))

    chunks = create_canary_chunks(path, 10.0)

    assert [(chunk.start_frame, chunk.frame_count) for chunk, _ in chunks] == [
        (0, 160000),
        (160000, 160000),
        (320000, 80000),
    ]
    assert chunks[-1][0].offset_seconds == pytest.approx(20.0)


def test_chunk_offsets_use_frame_indexes_not_accumulated_floats(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(b"\x00\x00" * (63 * SAMPLE_RATE))

    chunks = create_canary_chunks(path, 7.0)

    assert [(chunk.start_frame, chunk.frame_count) for chunk, _ in chunks] == [
        (index * 112000, 112000) for index in range(9)
    ]
    assert chunks[-1][0].offset_seconds == pytest.approx(56.0)


def test_write_canary_chunk_copies_the_exact_frame_range(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(b"\x00\x00" * (int(12.5 * SAMPLE_RATE)))
    second_chunk = create_canary_chunks(source, 10.0)[1][0]

    destination = write_canary_chunk(tmp_path, second_chunk, source)

    with wave.open(str(destination), "rb") as chunk_file:
        assert chunk_file.getframerate() == SAMPLE_RATE
        assert chunk_file.getnchannels() == 1
        assert chunk_file.getsampwidth() == 2
        assert chunk_file.getnframes() == int(2.5 * SAMPLE_RATE)
        chunk_file.setpos(0)


def test_transcribe_requires_normalized_pcm_characteristics(tmp_path: Path) -> None:
    stereo_path = tmp_path / "normalized.wav"
    with wave.open(str(stereo_path), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(b"\x00\x00\x00\x00" * 1600)
    model = Model([Hypothesis([{"word": "Hallo", "start": 0.0, "end": 0.1}])])
    instance = make_transcriber(tmp_path)
    instance._model = model

    with pytest.raises(ASROutputError, match="16 kHz mono 16-bit PCM"):
        instance.transcribe(NormalizedAudio(stereo_path, AudioMetadata("m.wav", 0.1)))


def test_chunk_files_are_removed_after_success(tmp_path: Path) -> None:
    model = Model([Hypothesis([{"word": "Hallo", "start": 0.0, "end": 0.3}])])
    instance = make_transcriber(tmp_path)
    instance._model = model

    instance.transcribe(pcm_audio(tmp_path, 3.0))

    assert chunk_file_paths(model)[0].startswith(str(tmp_path))
    assert list(tmp_path.glob("canary-chunks.*")) == []


def test_chunk_files_are_removed_after_model_failure(tmp_path: Path) -> None:
    class FailingModel:
        def transcribe(self, audio: list[str], **kwargs: object) -> list[Hypothesis]:
            raise RuntimeError("GPU went away")

    instance = make_transcriber(tmp_path)
    instance._model = FailingModel()

    with pytest.raises(ASROutputError, match="Canary recognition failed"):
        instance.transcribe(pcm_audio(tmp_path, 3.0))

    assert list(tmp_path.glob("canary-chunks.*")) == []


def test_chunk_files_are_removed_after_timestamp_parsing_failure(tmp_path: Path) -> None:
    model = Model([Hypothesis([{"word": "Hallo", "start": 0.3, "end": 0.2}])])
    instance = make_transcriber(tmp_path)
    instance._model = model

    with pytest.raises(ASROutputError, match="outside the chunk"):
        instance.transcribe(pcm_audio(tmp_path, 3.0))

    assert list(tmp_path.glob("canary-chunks.*")) == []


def test_validate_canary_words_rejects_timestamp_reset() -> None:
    words = [
        ASRWord(text="Erste", start=0.0, end=1.0, confidence=None),
        ASRWord(text="Zweite", start=0.2, end=0.5, confidence=None),
    ]

    with pytest.raises(ASROutputError, match="reset or reorder"):
        validate_canary_words(words)


def test_validate_canary_words_rejects_negative_timestamps() -> None:
    words = [ASRWord(text="Erste", start=-0.5, end=1.0, confidence=None)]

    with pytest.raises(ASROutputError, match="invalid"):
        validate_canary_words(words)


def test_validate_canary_words_allows_monotonic_chunk_sequence() -> None:
    validate_canary_words(
        [
            ASRWord(text="Erste", start=0.0, end=1.0, confidence=None),
            ASRWord(text="Zweite", start=11.0, end=12.0, confidence=None),
        ]
    )


def test_recognition_runner_preserves_prepared_sha_and_canary_metadata(tmp_path: Path) -> None:
    normalized = pcm_audio(tmp_path, 2.0)
    instance = make_transcriber(tmp_path)
    instance._model = Model([Hypothesis([{"word": "Hallo", "start": 0.0, "end": 0.5}])])
    prepared = PreparedRecording(
        audio=normalized,
        diarization=[DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
        work_directory=tmp_path,
        normalized_audio_sha256=sha256_file(normalized.path),
        diarization_model="pyannote/test",
        language="de-DE",
        cleanup_enabled=False,
    )

    result = RecognitionRunner().recognize(prepared, instance, "canary")

    assert result.metadata.backend == "canary"
    assert result.metadata.model == "nvidia/canary-1b-v2"
    assert result.metadata.normalized_audio_sha256 == prepared.normalized_audio_sha256
    assert result.metadata.runtime.name == "nemo"
    assert result.metadata.backend_configuration["source_language"] == "de"
    assert result.metadata.backend_configuration["target_language"] == "de"
    assert result.metadata.backend_configuration["inference_mode"] == (
        "sequential_non_overlapping_chunks"
    )


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
    with pytest.raises(ASROutputError, match="outside the chunk"):
        flatten_canary_words([Hypothesis([{"word": "Hallo", "start": 0.3, "end": 0.2}])])
    with pytest.raises(ASROutputError, match="outside the chunk"):
        flatten_canary_words([Hypothesis([{"word": "Hallo", "start": -1.0, "end": 0.2}])])


def test_flatten_canary_words_rebases_by_offset() -> None:
    words = flatten_canary_words(
        [Hypothesis([{"word": "Rebasiert", "start": 3.25, "end": 3.8}])],
        offset_seconds=70.0,
    )

    assert [(word.text, word.start, word.end) for word in words] == [("Rebasiert", 73.25, 73.8)]


def test_canary_import_is_lazy() -> None:
    import sys

    sys.modules.pop("nemo", None)
    sys.modules.pop("nemo.collections", None)
    assert "nemo" not in sys.modules

    CanaryTranscriber("nvidia/canary-1b-v2", "cuda", "de", 10.0)

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
    instance = CanaryTranscriber(str(artifact), "cuda", "de", 10.0)

    instance.load()

    assert calls == [(str(artifact), "cuda")]
    assert instance.backend_models["model_file"] == str(artifact)
    assert instance.runtime_provenance.name == "nemo"
    assert "torch" in instance.runtime_provenance.components
    assert "cuda" in instance.runtime_provenance.components


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


def test_frames_per_chunk_converts_duration_to_exact_frames() -> None:
    assert frames_per_chunk(10.0, 16000) == 160000
    assert frames_per_chunk(0.5, 16000) == 8000
    with pytest.raises(ValueError, match="positive"):
        frames_per_chunk(0.0, 16000)


def test_transcriber_rejects_nonpositive_chunk_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        CanaryTranscriber("nvidia/canary-1b-v2", "cuda", "de", 0.0)