"""Test-double wiring test and optional real-model smoke test, disabled in CI."""

import os
from pathlib import Path

import pytest

from speech_transcriber.config import PipelineConfig
from speech_transcriber.models import ASRWord, AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.preparation import PreparationRunner
from speech_transcriber.recognition import RecognitionRunner
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.factory import create_transcriber

real_models_only = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 and MODEL_TEST_AUDIO=/path/to/audio.wav to run real models",
)


class FakePreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def normalize(self, source: Path, destination: Path) -> AudioMetadata:
        self.calls += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"normalized audio")
        return AudioMetadata(source=source.name, duration_seconds=2.0)


class FakeDiarizer:
    def __init__(self) -> None:
        self.released = False
        self.calls = 0

    def diarize(self, normalized_wav: Path) -> list[DiarizationSegment]:
        self.calls += 1
        return [DiarizationSegment("SPEAKER_00", 0.0, 2.0)]

    def release(self) -> None:
        self.released = True


class FakeTranscriber:
    def __init__(self) -> None:
        self.released = False
        self.loaded = False
        self.device = "cpu"
        self.model_reference = "fake/asr"
        self.dtype_name = "float32"
        self.capabilities = TranscriberCapabilities(True, True, True, True)
        self.audio: NormalizedAudio | None = None

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        self.audio = audio
        return [ASRWord("hallo", 0.5, start=0.0)]

    def release(self) -> None:
        self.released = True


def test_split_stage_runners_cross_the_prepared_artifact_boundary(tmp_path: Path) -> None:
    diarizer = FakeDiarizer()
    transcriber = FakeTranscriber()
    config = PipelineConfig(
        input_path=tmp_path / "audio.wav",
        output_directory=tmp_path / "result",
        working_directory=tmp_path / "work",
        asr_backend="parakeet",
    )

    prepared = PreparationRunner(
        config,
        diarizer_factory=lambda: diarizer,
        preprocessor=FakePreprocessor(),
    ).prepare()
    recognition = RecognitionRunner().recognize(prepared, transcriber, "parakeet")

    assert [word.text for word in recognition.words] == ["hallo"]
    assert diarizer.released and transcriber.loaded and transcriber.released
    assert recognition.metadata.normalized_audio_sha256 == prepared.normalized_audio_sha256


@real_models_only
def test_real_model_recognition_and_finalization(tmp_path: Path) -> None:
    source = os.environ.get("MODEL_TEST_AUDIO")
    if not source:
        pytest.skip("MODEL_TEST_AUDIO is not configured")
    config = PipelineConfig(
        input_path=Path(source),
        output_directory=tmp_path / "result",
        working_directory=tmp_path / "work",
        asr_backend=os.environ.get("MODEL_TEST_BACKEND", "parakeet"),
        parakeet_segment_duration=30,
        parakeet_segment_overlap=5,
        qwen_segment_duration=30,
        qwen_segment_overlap=5,
    )
    from speech_transcriber.prepared import write_prepared_recording

    runner = PreparationRunner.create_default(config)
    prepared = runner.prepare()
    prepared_artifact = tmp_path / "prepared"
    write_prepared_recording(prepared, prepared_artifact)
    loaded = __import__(
        "speech_transcriber.prepared", fromlist=["load_prepared_recording"]
    ).load_prepared_recording(prepared_artifact)
    from speech_transcriber.recognition import RecognitionRunner

    device = resolve_device_for_test(config)
    recognition = RecognitionRunner().recognize(
        loaded, create_transcriber(config, device), config.asr_backend
    )
    assert recognition.words
    if config.asr_backend == "voxtral":
        assert all(word.start is None and word.end >= 0 for word in recognition.words)
    else:
        assert all(word.start is not None and word.end >= word.start for word in recognition.words)


def resolve_device_for_test(config: object) -> str:
    from speech_transcriber.runtime.device import resolve_device

    return resolve_device("cpu")