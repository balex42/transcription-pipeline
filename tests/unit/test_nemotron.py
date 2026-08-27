from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from meeting_transcriber.errors import NemotronStreamingError
from meeting_transcriber.models import AudioMetadata, NormalizedAudio
from meeting_transcriber.transcription.nemotron import (
    NemotronTranscriber,
    aggregate_nemotron_tokens,
)


class Inputs(dict[str, object]):
    def to(self, device: str, dtype: object) -> Inputs:
        self["device"] = device
        self["dtype"] = dtype
        return self


class Processor:
    default_num_lookahead_tokens = 3
    num_samples_first_audio_chunk = 4
    num_samples_per_audio_chunk = 4
    num_mel_frames_first_audio_chunk = 2
    num_mel_frames_per_audio_chunk = 1
    streaming_latency_ms = 320

    class FeatureExtractor:
        hop_length = 4
        n_fft = 8

    feature_extractor = FeatureExtractor()

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, audio: object, **kwargs: object) -> Inputs:
        self.calls.append({"audio": audio, **kwargs})
        return Inputs(
            input_features=f"feature-{len(self.calls)}",
            num_lookahead_tokens=self.default_num_lookahead_tokens,
            prompt_ids="de-DE",
        )

    def set_num_lookahead_tokens(self, value: int) -> None:
        if value not in {0, 3, 6, 13}:
            raise ValueError("unsupported")
        self.default_num_lookahead_tokens = value

    def decode(
        self, sequences: object, **kwargs: object
    ) -> tuple[str, list[list[dict[str, object]]]]:
        assert kwargs == {"durations": "durations", "skip_special_tokens": True}
        return "Guten Morgen,", [[
            {"token": " Gu", "start": 0.0, "end": 0.08},
            {"token": "ten", "start": 0.08, "end": 0.16},
            {"token": " Morgen", "start": 0.16, "end": 0.24},
            {"token": ",", "start": 0.24, "end": 0.32},
        ]]


class Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> object:
        features = list(kwargs["input_features"])
        self.calls.append({**kwargs, "input_features": features})
        return type("Output", (), {"sequences": "sequences", "durations": "durations"})()


def audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "meeting.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(12, dtype="<i2").tobytes())
    return NormalizedAudio(path, AudioMetadata(path.name, 1.0))


def test_nemotron_uses_one_cache_aware_generation_for_all_stream_buffers(tmp_path: Path) -> None:
    processor = Processor()
    model = Model()
    transcriber = NemotronTranscriber("/models/nemotron", "cpu", lookahead=6)
    transcriber._processor = processor
    transcriber._model = model

    words = transcriber.transcribe(audio(tmp_path))

    assert [(word.text, word.start, word.end) for word in words] == [
        ("Guten", 0.0, 0.16),
        ("Morgen,", 0.16, 0.32),
    ]
    assert len(model.calls) == 1
    assert model.calls[0]["input_features"] == ["feature-1", "feature-2", "feature-3"]
    assert [call["is_first_audio_chunk"] for call in processor.calls] == [True, False, False]
    assert all(
        call["is_streaming"] is True and call["language"] == "de-DE" for call in processor.calls
    )
    assert transcriber.backend_configuration["num_lookahead_tokens"] == 6
    assert transcriber.backend_metrics["stream_buffers_processed"] == 3.0


def test_nemotron_resets_streaming_state_for_each_meeting(tmp_path: Path) -> None:
    transcriber = NemotronTranscriber("/models/nemotron", "cpu")
    transcriber._processor = Processor()
    transcriber._model = Model()

    transcriber.transcribe(audio(tmp_path))
    transcriber.transcribe(audio(tmp_path))

    assert transcriber.backend_metrics["stream_buffers_processed"] == 3.0
    assert len(transcriber._model.calls) == 2  # type: ignore[union-attr]


def test_nemotron_rejects_unsupported_lookahead(tmp_path: Path) -> None:
    transcriber = NemotronTranscriber("/models/nemotron", "cpu", lookahead=99)
    transcriber._processor = Processor()
    transcriber._model = Model()
    with pytest.raises(NemotronStreamingError, match="does not support"):
        transcriber.transcribe(audio(tmp_path))


def test_token_aggregation_handles_german_punctuation_and_subwords() -> None:
    words = aggregate_nemotron_tokens(
        [
            {"token": " Mül", "start": 0.0, "end": 0.08},
            {"token": "ler", "start": 0.08, "end": 0.16},
            {"token": " fährt", "start": 0.16, "end": 0.24},
            {"token": " Open", "start": 0.24, "end": 0.32},
            {"token": "Shift", "start": 0.32, "end": 0.4},
            {"token": "-", "start": 0.4, "end": 0.48},
            {"token": "Cluster", "start": 0.48, "end": 0.56},
            {"token": " Straße", "start": 0.56, "end": 0.64},
            {"token": " 3", "start": 0.64, "end": 0.72},
            {"token": ",", "start": 0.72, "end": 0.8},
            {"token": "5", "start": 0.8, "end": 0.88},
            {"token": ".", "start": 0.88, "end": 0.96},
            {"token": "<blank>", "start": 0.96, "end": 1.04},
        ],
        2.0,
    )
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Müller", 0.0, 0.16),
        ("fährt", 0.16, 0.24),
        ("OpenShift-Cluster", 0.24, 0.56),
        ("Straße", 0.56, 0.64),
        ("3,5.", 0.64, 0.96),
    ]


def test_token_aggregation_rejects_non_monotonic_timestamps() -> None:
    with pytest.raises(NemotronStreamingError, match="non-monotonic"):
        aggregate_nemotron_tokens(
            [
                {"token": " eins", "start": 0.2, "end": 0.3},
                {"token": " zwei", "start": 0.1, "end": 0.2},
            ],
            1.0,
        )
