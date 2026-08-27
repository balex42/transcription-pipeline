from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

from meeting_transcriber.models import AudioMetadata, NormalizedAudio
from meeting_transcriber.transcription.parakeet import ParakeetTranscriber
from meeting_transcriber.transcription.whisper import WhisperTranscriber


class Inputs(dict[str, object]):
    def to(self, device: str, dtype: object) -> Inputs:
        return self


class ParakeetProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> Inputs:
        self.calls += 1
        return Inputs()

    def decode(self, token_ids: object, **kwargs: object) -> tuple[str, list[dict[str, object]]]:
        return "", [{"word": "Wort", "start": 0.0, "end": 0.5}]


class ParakeetModel:
    def generate(self, **kwargs: object) -> object:
        return type(
            "Output", (), {"sequences": torch.tensor([[1]]), "durations": torch.tensor([[1]])}
        )()


class WhisperPipeline:
    def __init__(self) -> None:
        self.requests: list[tuple[object, dict[str, object]]] = []

    def __call__(self, inputs: object, **kwargs: object) -> dict[str, object]:
        self.requests.append((inputs, kwargs))
        start = 12.0 if len(self.requests) > 1 else 0.0
        return {"chunks": [{"text": " Guten", "timestamp": (start, start + 0.5)}]}


def audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "meeting.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(48_000, dtype="<i2").tobytes())
    return NormalizedAudio(path, AudioMetadata(path.name, 3.0))


def test_parakeet_segments_and_reconciles_inside_its_adapter(tmp_path: Path) -> None:
    processor = ParakeetProcessor()
    transcriber = ParakeetTranscriber("/models/parakeet", "cpu", 2.0, 1.0)
    transcriber._processor = processor
    transcriber._model = ParakeetModel()

    words = transcriber.transcribe(audio(tmp_path))

    assert processor.calls == 2
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Wort", 0.0, 0.5),
        ("Wort", 1.0, 1.5),
    ]


def test_whisper_transcribes_model_window_segments_and_reconciles_them(
    tmp_path: Path,
) -> None:
    pipeline = WhisperPipeline()
    transcriber = WhisperTranscriber("/models/whisper", "cpu")
    transcriber._pipeline = pipeline

    words = transcriber.transcribe(audio(tmp_path))

    assert len(pipeline.requests) == 1
    samples, kwargs = pipeline.requests[0]
    assert isinstance(samples, np.ndarray) and len(samples) == 48_000
    assert "chunk_length_s" not in kwargs
    assert "stride_length_s" not in kwargs
    assert kwargs["return_timestamps"] == "word"
    assert transcriber.backend_configuration == {
        "segment_duration_seconds": 30.0,
        "segment_overlap_seconds": 5.0,
    }
    assert transcriber.backend_metrics["segments_processed"] == 1.0
    assert [(word.text, word.start, word.end) for word in words] == [("Guten", 0.0, 0.5)]


def test_whisper_reconciles_words_across_overlapping_model_windows(tmp_path: Path) -> None:
    path = tmp_path / "long-meeting.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(1_040_000, dtype="<i2").tobytes())
    pipeline = WhisperPipeline()
    transcriber = WhisperTranscriber("/models/whisper", "cpu")
    transcriber._pipeline = pipeline

    words = transcriber.transcribe(NormalizedAudio(path, AudioMetadata(path.name, 65.0)))

    assert [len(request[0]) for request in pipeline.requests] == [480_000, 480_000, 240_000]
    assert transcriber.backend_metrics["segments_processed"] == 3.0
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Guten", 0.0, 0.5),
        ("Guten", 37.0, 37.5),
        ("Guten", 62.0, 62.5),
    ]
