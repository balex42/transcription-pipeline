import json
import shutil
from pathlib import Path

import pytest

from speech_transcriber.asr_artifact import load_asr_recognition, write_asr_recognition
from speech_transcriber.config import PipelineConfig
from speech_transcriber.models import (
    ASRRecognitionResult,
    ASRRunMetadata,
    ASRWord,
    AudioMetadata,
    DiarizationSegment,
    NormalizedAudio,
    RuntimeProvenance,
)
from speech_transcriber.pipeline import PreparedRecording, TranscriptionPipeline


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
    assert payload["schema_version"] == 1
    assert payload["asr_words_file"] == "asr_words.json"
    assert "/models" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("filename", "mutate", "message"),
    [
        ("metadata.json", lambda value: value.update(schema_version=2), "unsupported ASR artifact"),
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


def test_finalization_uses_only_prepared_and_recognition_artifacts(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    audio.write_bytes(b"normalized audio")
    prepared = PreparedRecording(
        audio=NormalizedAudio(audio, AudioMetadata("meeting.wav", 2.0)),
        diarization=[DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
        work_directory=tmp_path,
        diarization_model="pyannote/test",
        language="de-DE",
        cleanup_enabled=False,
    )
    recognition_directory = tmp_path / "asr"
    write_asr_recognition(_recognition(), recognition_directory)

    def no_transcriber() -> object:
        raise AssertionError("finalization must not instantiate a Transcriber")

    pipeline = TranscriptionPipeline(
        PipelineConfig(audio, tmp_path / "result", tmp_path / "work"),
        diarizer_factory=lambda: None,  # type: ignore[arg-type]
        transcriber_factory=no_transcriber,  # type: ignore[arg-type]
    )
    result = pipeline.finalize_prepared(
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
