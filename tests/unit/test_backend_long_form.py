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
        self.request: tuple[object, dict[str, object]] | None = None

    def __call__(self, inputs: object, **kwargs: object) -> dict[str, object]:
        self.request = inputs, kwargs
        return {"chunks": [{"text": " Guten", "timestamp": (0.0, 0.5)}]}


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


def test_whisper_receives_the_complete_recording_for_native_long_form(tmp_path: Path) -> None:
    pipeline = WhisperPipeline()
    transcriber = WhisperTranscriber("/models/whisper", "cpu")
    transcriber._pipeline = pipeline

    words = transcriber.transcribe(audio(tmp_path))

    assert pipeline.request is not None
    samples, kwargs = pipeline.request
    assert isinstance(samples, np.ndarray) and len(samples) == 48_000
    assert kwargs["chunk_length_s"] == 30.0
    assert kwargs["stride_length_s"] == (5.0, 5.0)
    assert transcriber.backend_configuration == {
        "long_form_strategy": "chunked",
        "chunk_length_seconds": 30.0,
        "stride_length_seconds": 5.0,
    }
    assert [(word.text, word.start, word.end) for word in words] == [("Guten", 0.0, 0.5)]
