"""Real-model smoke test, intentionally disabled in normal CI."""

import os
from pathlib import Path

import pytest

from meeting_transcriber.comparison import ASRComparisonRunner
from meeting_transcriber.config import PipelineConfig
from meeting_transcriber.models import ASRWord, AudioMetadata, DiarizationSegment, NormalizedAudio
from meeting_transcriber.pipeline import MeetingTranscriptionPipeline, create_default_pipeline
from meeting_transcriber.transcription.base import TranscriberCapabilities

real_models_only = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 and MODEL_TEST_AUDIO=/path/to/German.wav to run real models",
)


class FakePreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def normalize(self, source: Path, destination: Path) -> AudioMetadata:
        self.calls += 1
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


def test_pipeline_with_model_test_doubles(tmp_path: Path) -> None:
    diarizer = FakeDiarizer()
    transcriber = FakeTranscriber()
    config = PipelineConfig(
        input_path=tmp_path / "meeting.wav",
        output_directory=tmp_path / "result",
        working_directory=tmp_path / "work",
    )
    pipeline = MeetingTranscriptionPipeline(
        config,
        diarizer_factory=lambda: diarizer,
        transcriber_factory=lambda: transcriber,
        preprocessor=FakePreprocessor(),
    )
    result = pipeline.run()
    assert [(word.text, word.speaker) for word in result.transcript.words] == [
        ("hallo", "SPEAKER_00")
    ]
    assert diarizer.released and transcriber.loaded and transcriber.released
    assert transcriber.audio is not None
    assert (config.output_directory / "transcript.json").is_file()
    assert not config.working_directory.exists()


def test_comparison_prepares_once_and_separates_backend_outputs(tmp_path: Path) -> None:
    preprocessor = FakePreprocessor()
    diarizer = FakeDiarizer()
    transcribers: list[FakeTranscriber] = []
    config = PipelineConfig(tmp_path / "meeting.wav", tmp_path / "comparison", tmp_path / "work")
    pipeline = MeetingTranscriptionPipeline(
        config,
        diarizer_factory=lambda: diarizer,
        transcriber_factory=FakeTranscriber,
        preprocessor=preprocessor,
    )

    def build_transcriber(_: PipelineConfig, __: str) -> FakeTranscriber:
        transcriber = FakeTranscriber()
        transcribers.append(transcriber)
        return transcriber

    ASRComparisonRunner(pipeline, build_transcriber).run(
        ["parakeet", "whisper", "qwen", "nemotron"], config.output_directory
    )
    assert (preprocessor.calls, diarizer.calls, len(transcribers)) == (1, 1, 4)
    assert (config.output_directory / "diarization.json").is_file()
    assert (config.output_directory / "metadata.json").is_file()
    for backend in ("parakeet", "whisper", "qwen", "nemotron"):
        assert (config.output_directory / backend / "transcript.json").is_file()
        assert (config.output_directory / backend / "asr_words.json").is_file()
    assert (config.output_directory / "qwen" / "metadata.json").is_file()


@real_models_only
def test_real_model_pipeline(tmp_path: Path) -> None:
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
    result = create_default_pipeline(config).run()
    assert (result.output_directory / "transcript.json").is_file()  # type: ignore[operator]
    assert result.asr_words
    assert all(word.start is not None and word.end >= word.start for word in result.asr_words)
