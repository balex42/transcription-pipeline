from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_transcriber.errors import CohereRecognitionError
from speech_transcriber.models import AudioSegment
from speech_transcriber.transcription.cohere.recognizer import CohereRecognizer


class Inputs(dict[str, object]):
    def to(self, device: str, dtype: object) -> Inputs:
        self["device"] = device
        self["dtype"] = dtype
        return self


class Processor:
    def __init__(self, decoded: object = "Guten Morgen") -> None:
        self.decoded = decoded
        self.request: dict[str, object] | None = None
        self.decode_request: dict[str, object] | None = None

    def __call__(self, audio: object, **kwargs: object) -> Inputs:
        self.request = {"audio": audio, **kwargs}
        return Inputs(audio_chunk_index=[(0, None)])

    def decode(self, token_ids: object, **kwargs: object) -> object:
        self.decode_request = kwargs
        return self.decoded


class Model:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3]])


def segment() -> AudioSegment:
    return AudioSegment(7, 20.0, 22.0, np.zeros(32_000, dtype=np.float32))


def test_recognizer_requests_deterministic_transcription() -> None:
    recognizer = CohereRecognizer("/models/cohere", "cpu")
    processor = Processor()
    model = Model()
    recognizer._processor = processor
    recognizer._model = model

    assert recognizer.recognize(segment()) == "Guten Morgen"
    assert processor.request is not None
    assert processor.request["sampling_rate"] == 16_000
    assert processor.request["language"] == "de"
    assert processor.request["punctuation"] is True
    assert processor.decode_request == {
        "skip_special_tokens": True,
        "audio_chunk_index": [(0, None)],
        "language": "de",
    }
    assert model.kwargs is not None
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["num_beams"] == 1
    assert model.kwargs["max_new_tokens"] == 256


def test_recognizer_reduces_locale_to_base_language_code() -> None:
    recognizer = CohereRecognizer("/models/cohere", "cpu", language="en-US")
    processor = Processor()
    recognizer._processor = processor
    recognizer._model = Model()
    recognizer.recognize(segment())
    assert processor.request is not None
    assert processor.request["language"] == "en"


def test_recognizer_accepts_reassembled_single_item_decode() -> None:
    recognizer = CohereRecognizer("/models/cohere", "cpu")
    recognizer._processor = Processor([" Guten Morgen "])
    recognizer._model = Model()
    assert recognizer.recognize(segment()) == "Guten Morgen"


def test_recognizer_rejects_unexpected_decode_shape() -> None:
    recognizer = CohereRecognizer("/models/cohere", "cpu")
    recognizer._processor = Processor(["eins", "zwei"])
    recognizer._model = Model()
    with pytest.raises(CohereRecognitionError, match="unsupported decoded response"):
        recognizer.recognize(segment())


def test_recognizer_release_drops_model_resources() -> None:
    recognizer = CohereRecognizer("/models/cohere", "cpu")
    recognizer._processor = Processor()
    recognizer._model = Model()
    recognizer.release()
    assert recognizer._processor is None
    assert recognizer._model is None
