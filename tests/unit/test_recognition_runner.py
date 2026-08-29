from __future__ import annotations

from pathlib import Path

from speech_transcriber.models import (
    ASRWord,
    AudioMetadata,
    DiarizationSegment,
    NormalizedAudio,
    RuntimeProvenance,
)
from speech_transcriber.prepared import PreparedRecording, sha256_file
from speech_transcriber.recognition import RecognitionRunner
from speech_transcriber.transcription.base import TranscriberCapabilities


class FakeTranscriber:
    def __init__(self) -> None:
        self.device = "cuda"
        self.dtype_name = "float16"
        self.model_reference = "Systran/faster-whisper-large-v3"
        self.capabilities = TranscriberCapabilities(True, True, True, True)
        self.loaded = False
        self.released = False
        self.runtime_provenance = RuntimeProvenance(
            name="faster-whisper",
            version="1.2.1",
            components={"ctranslate2": "4.8.1", "huggingface_hub": "1.28.0"},
        )
        self.backend_metrics = {"detected_language_probability": 0.98}
        self.backend_models = {"model_path": "/models/huggingface/hub/snapshot"}
        self.backend_configuration = {
            "language": "de",
            "compute_type": "float16",
            "word_timestamps": True,
            "vad_filter": False,
            "beam_size": 5,
        }

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, _: NormalizedAudio) -> list[ASRWord]:
        return [ASRWord("hallo", end=0.5, start=0.0, confidence=0.99)]

    def release(self) -> None:
        self.released = True


class MemoryMetrics:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.peak_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def peak(self) -> tuple[int | None, int | None]:
        self.peak_calls += 1
        return 1024, 2048


def prepared(tmp_path: Path) -> PreparedRecording:
    audio = tmp_path / "normalized.wav"
    audio.write_bytes(b"normalized audio")
    return PreparedRecording(
        audio=NormalizedAudio(audio, AudioMetadata("meeting.wav", 2.0)),
        diarization=[DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
        work_directory=tmp_path,
        normalized_audio_sha256=sha256_file(audio),
        diarization_model="pyannote/test",
        language="de-DE",
        cleanup_enabled=False,
    )


def test_recognition_runner_produces_canonical_artifact_metadata(tmp_path: Path) -> None:
    transcriber = FakeTranscriber()
    metrics = MemoryMetrics()
    result = RecognitionRunner(metrics).recognize(prepared(tmp_path), transcriber, "faster-whisper")

    assert transcriber.loaded and transcriber.released
    assert metrics.reset_calls == 1 and metrics.peak_calls == 1
    assert [(word.text, word.start, word.end, word.confidence) for word in result.words] == [
        ("hallo", 0.0, 0.5, 0.99)
    ]
    metadata = result.metadata
    assert metadata.backend == "faster-whisper"
    assert metadata.model == "Systran/faster-whisper-large-v3"
    assert metadata.device == "cuda"
    assert metadata.dtype == "float16"
    assert metadata.audio_duration_seconds == 2.0
    assert metadata.peak_cuda_memory_allocated_bytes == 1024
    assert metadata.peak_cuda_memory_reserved_bytes == 2048
    assert metadata.normalized_audio_sha256 == prepared(tmp_path).normalized_audio_sha256
    assert metadata.runtime.name == "faster-whisper"
    assert metadata.runtime.components["ctranslate2"] == "4.8.1"
    assert metadata.backend_metrics == {"detected_language_probability": 0.98}
    assert metadata.backend_configuration["beam_size"] == 5


def test_recognition_runner_without_memory_metrics_records_none(tmp_path: Path) -> None:
    transcriber = FakeTranscriber()
    result = RecognitionRunner().recognize(prepared(tmp_path), transcriber, "faster-whisper")

    assert result.metadata.peak_cuda_memory_allocated_bytes is None
    assert result.metadata.peak_cuda_memory_reserved_bytes is None


def test_recognition_runner_releases_transcriber_on_failure(tmp_path: Path) -> None:
    class FailingTranscriber(FakeTranscriber):
        def transcribe(self, _: NormalizedAudio) -> list[ASRWord]:
            raise RuntimeError("model failed")

    transcriber = FailingTranscriber()
    try:
        RecognitionRunner().recognize(prepared(tmp_path), transcriber, "faster-whisper")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected recognition failure")

    assert transcriber.loaded and transcriber.released
