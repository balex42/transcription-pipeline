from __future__ import annotations

import numpy as np
import torch

from meeting_transcriber.models import AudioChunk
from meeting_transcriber.transcription.qwen.recognizer import QwenRecognizer


class Inputs(dict[str, object]):
    def to(self, device: str, dtype: object) -> Inputs:
        self["device"] = device
        self["dtype"] = dtype
        return self


class Processor:
    def __init__(self, text: str = "Guten Morgen") -> None:
        self.text = text
        self.request: dict[str, object] | None = None

    def apply_transcription_request(self, **kwargs: object) -> Inputs:
        self.request = kwargs
        return Inputs(input_ids=torch.tensor([[1, 2]]))

    def decode(self, token_ids: object, **kwargs: object) -> list[dict[str, str]]:
        assert kwargs == {"return_format": "parsed"}
        return [{"transcription": self.text}]


class Model:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3]])


def chunk() -> AudioChunk:
    return AudioChunk(7, 20.0, 22.0, np.zeros(32_000, dtype=np.float32))


def test_recognizer_requests_deterministic_german_transcription() -> None:
    recognizer = QwenRecognizer("/models/qwen", "cpu", context="Names: Fenske")
    processor = Processor()
    model = Model()
    recognizer._processor = processor
    recognizer._model = model

    assert recognizer.recognize(chunk()) == "Guten Morgen"
    assert processor.request is not None
    assert processor.request["language"] == "de"
    assert processor.request["prompt"] == "Names: Fenske"
    assert model.kwargs is not None
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["num_beams"] == 1
    assert model.kwargs["max_new_tokens"] == 2_048


def test_recognizer_preserves_empty_transcript_for_chunk_orchestration() -> None:
    recognizer = QwenRecognizer("/models/qwen", "cpu")
    recognizer._processor = Processor("  ")
    recognizer._model = Model()
    assert recognizer.recognize(chunk()) == ""
