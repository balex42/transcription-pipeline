import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from speech_transcriber.asr_artifact import load_asr_recognition, write_asr_recognition
from speech_transcriber.config import PipelineConfig
from speech_transcriber.finalization import TranscriptFinalizer
from speech_transcriber.models import (
    ASRRecognitionResult,
    ASRRunMetadata,
    ASRWord,
    AudioMetadata,
    DiarizationSegment,
    NormalizedAudio,
    RuntimeProvenance,
)
from speech_transcriber.prepared import PreparedRecording, sha256_file


def _recognition() -> ASRRecognitionResult:
    return ASRRecognitionResult(
        words=[
            ASRWord("hallo", end=0.5, start=0.0, confidence=0.9),
            ASRWord("welt", end=1.0, start=None),
        ],
        metadata=ASRRunMetadata(
            backend="parakeet",
            model="/models/parakeet",
            device="cuda",
            dtype="float16",
            audio_duration_seconds=2.0,
            model_load_seconds=1.0,
            transcription_seconds=0.5,
            total_asr_seconds=1.5,
            real_time_factor=0.75,
            peak_cuda_memory_allocated_bytes=42,
            peak_cuda_memory_reserved_bytes=84,
            normalized_audio_sha256="a" * 64,
            runtime=RuntimeProvenance(
                name="transformers",
                version="5.13.0",
                components={"torch": "2.8.0", "transformers": "5.13.0"},
            ),
            backend_models={"model": "/models/parakeet"},
            backend_configuration={"local_model": "/models/parakeet", "batch_size": 1},
        ),
    )


def test_recognition_artifact_round_trip_is_relocatable_and_redacts_absolute_paths(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    write_asr_recognition(_recognition(), original)
    relocated = tmp_path / "relocated"
    shutil.move(str(original), relocated)

    loaded = load_asr_recognition(relocated)

    assert loaded.words[1].start is None
    assert loaded.metadata.runtime.name == "transformers"
    assert loaded.metadata.model == "parakeet"
    payload = json.loads((relocated / "metadata.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["asr_words_file"] == "asr_words.json"
    assert "/models" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("filename", "mutate", "message"),
    [
        ("metadata.json", lambda value: value.update(schema_version=3), "unsupported ASR artifact"),
        (
            "asr_words.json",
            lambda value: value[0].update(speaker="SPEAKER_00"),
            "unsupported fields",
        ),
        ("asr_words.json", lambda value: value[0].update(start=1.0, end=0.5), "must not precede"),
    ],
)
def test_recognition_artifact_rejects_invalid_schema(
    tmp_path: Path, filename: str, mutate: object, message: str
) -> None:
    directory = tmp_path / "asr"
    write_asr_recognition(_recognition(), directory)
    path = directory / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)  # type: ignore[operator]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_asr_recognition(directory)


def test_recognition_artifact_validates_backend_fingerprint_metrics_and_paths(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "asr"
    recognition = _recognition()
    recognition = ASRRecognitionResult(
        words=recognition.words,
        metadata=replace(recognition.metadata, backend_metrics={"log_probability": -2.5}),
    )
    write_asr_recognition(recognition, directory)

    assert load_asr_recognition(directory).metadata.backend_metrics == {"log_probability": -2.5}
    with pytest.raises(ValueError, match="does not match"):
        load_asr_recognition(directory, expected_backend="qwen")
    with pytest.raises(ValueError, match="SHA-256"):
        load_asr_recognition(directory, expected_normalized_audio_sha256="b" * 64)

    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["backend_configuration"]["external_model"] = "/models/external"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="absolute path"):
        load_asr_recognition(directory)


def test_finalization_uses_only_prepared_and_recognition_artifacts(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    audio.write_bytes(b"normalized audio")
    prepared = PreparedRecording(
        audio=NormalizedAudio(audio, AudioMetadata("meeting.wav", 2.0)),
        diarization=[DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
        work_directory=tmp_path,
        normalized_audio_sha256=sha256_file(audio),
        diarization_model="pyannote/test",
        language="de-DE",
        cleanup_enabled=False,
    )
    recognition_directory = tmp_path / "asr"
    recognition = _recognition()
    recognition = ASRRecognitionResult(
        words=recognition.words,
        metadata=replace(recognition.metadata, normalized_audio_sha256=sha256_file(audio)),
    )
    write_asr_recognition(recognition, recognition_directory)
    result = TranscriptFinalizer(
        PipelineConfig(audio, tmp_path / "result", tmp_path / "work")
    ).finalize_prepared(
        prepared,
        load_asr_recognition(recognition_directory),
        tmp_path / "result",
    )

    assert [(word.text, word.speaker) for word in result.transcript.words] == [
        ("hallo", "SPEAKER_00"),
        ("welt", "SPEAKER_00"),
    ]
    assert result.transcript.asr_backend == "parakeet"
    assert result.transcript.asr_model == "parakeet"
    assert (tmp_path / "result" / "transcript.json").is_file()


def test_finalization_rejects_a_recognition_for_another_recording(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    audio.write_bytes(b"normalized audio")
    prepared = PreparedRecording(
        audio=NormalizedAudio(audio, AudioMetadata("meeting.wav", 2.0)),
        diarization=[],
        work_directory=tmp_path,
        normalized_audio_sha256=sha256_file(audio),
        diarization_model="pyannote/test",
        language="de-DE",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        config = PipelineConfig(audio, tmp_path / "result", tmp_path / "work")
        finalizer = TranscriptFinalizer(config)
        finalizer.finalize_prepared(
            prepared,
            _recognition(),
            tmp_path / "result",
        )


def test_finalizer_accepts_a_faster_whisper_artifact_without_backend_specific_logic(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "normalized.wav"
    audio.write_bytes(b"normalized audio")
    prepared = PreparedRecording(
        audio=NormalizedAudio(audio, AudioMetadata("meeting.wav", 2.0)),
        diarization=[DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
        work_directory=tmp_path,
        normalized_audio_sha256=sha256_file(audio),
        diarization_model="pyannote/test",
        language="de-DE",
        cleanup_enabled=False,
    )
    recognition_directory = tmp_path / "asr"
    recognition = ASRRecognitionResult(
        words=[
            ASRWord("hallo", end=0.5, start=0.0, confidence=0.99),
            ASRWord("welt", end=1.0, start=0.5, confidence=0.95),
        ],
        metadata=ASRRunMetadata(
            backend="faster-whisper",
            model="Systran/faster-whisper-large-v3",
            device="cuda",
            dtype="float16",
            audio_duration_seconds=2.0,
            model_load_seconds=1.0,
            transcription_seconds=0.5,
            total_asr_seconds=1.5,
            real_time_factor=0.75,
            peak_cuda_memory_allocated_bytes=None,
            peak_cuda_memory_reserved_bytes=None,
            normalized_audio_sha256=sha256_file(audio),
            transformers_version="unknown",
            torch_version="unknown",
            runtime=RuntimeProvenance(
                name="faster-whisper",
                version="1.2.1",
                components={"ctranslate2": "4.8.1", "huggingface_hub": "1.28.0"},
            ),
            backend_configuration={
                "language": "de",
                "compute_type": "float16",
                "word_timestamps": True,
                "vad_filter": False,
            },
        ),
    )
    write_asr_recognition(recognition, recognition_directory)
    result = TranscriptFinalizer(
        PipelineConfig(audio, tmp_path / "result", tmp_path / "work")
    ).finalize_prepared(
        prepared,
        load_asr_recognition(recognition_directory),
        tmp_path / "result",
        expected_backend="faster-whisper",
    )

    assert [(word.text, word.speaker) for word in result.transcript.words] == [
        ("hallo", "SPEAKER_00"),
        ("welt", "SPEAKER_00"),
    ]
    assert result.transcript.asr_backend == "faster-whisper"
    assert result.transcript.asr_model == "Systran/faster-whisper-large-v3"
    assert (tmp_path / "result" / "transcript.json").is_file()


def test_finalizer_rejects_a_faster_whisper_artifact_for_another_backend(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "normalized.wav"
    audio.write_bytes(b"normalized audio")
    prepared = PreparedRecording(
        audio=NormalizedAudio(audio, AudioMetadata("meeting.wav", 2.0)),
        diarization=[],
        work_directory=tmp_path,
        normalized_audio_sha256=sha256_file(audio),
        diarization_model="pyannote/test",
        language="de-DE",
    )
    recognition = ASRRecognitionResult(
        words=[ASRWord("hallo", end=0.5, start=0.0)],
        metadata=ASRRunMetadata(
            backend="faster-whisper",
            model="Systran/faster-whisper-large-v3",
            device="cuda",
            dtype="float16",
            audio_duration_seconds=2.0,
            model_load_seconds=1.0,
            transcription_seconds=0.5,
            total_asr_seconds=1.5,
            real_time_factor=0.75,
            peak_cuda_memory_allocated_bytes=None,
            peak_cuda_memory_reserved_bytes=None,
            normalized_audio_sha256=sha256_file(audio),
        ),
    )

    with pytest.raises(ValueError, match="does not match"):
        TranscriptFinalizer(
            PipelineConfig(audio, tmp_path / "result", tmp_path / "work")
        ).finalize_prepared(
            prepared,
            recognition,
            tmp_path / "result",
            expected_backend="parakeet",
        )
