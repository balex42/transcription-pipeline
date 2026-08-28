from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from speech_transcriber.errors import VoxtralTimestampError
from speech_transcriber.models import AudioMetadata, NormalizedAudio
from speech_transcriber.transcription.voxtral.timestamps import parse_voxtral_words
from speech_transcriber.transcription.voxtral.transcriber import VoxtralTranscriber


class Inputs(dict[str, object]):
    def to(self, device: str, dtype: object) -> Inputs:
        self["device"] = device
        self["dtype"] = dtype
        return self


class Tokenizer:
    pieces = {
        1: "[TRANSCRIBE]",
        2: "[STREAMING_PAD]",
        10: "Hallo",
        11: "Welt",
        99: "[STREAMING_WORD]",
    }

    def convert_ids_to_tokens(self, ids: list[int], skip_special_tokens: bool = False) -> list[str]:
        return [self.pieces[token] for token in ids]


class Processor:
    num_right_pad_tokens = 1
    raw_audio_length_per_tok = 4
    num_samples_first_audio_chunk = 4
    num_samples_per_audio_chunk = 4
    num_mel_frames_first_audio_chunk = 2
    audio_length_per_tok = 1
    num_delay_tokens = 1
    tokenizer = Tokenizer()

    class FeatureExtractor:
        hop_length = 4
        win_length = 8

    feature_extractor = FeatureExtractor()

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, audio: object, **kwargs: object) -> Inputs:
        self.calls.append({"audio": audio, **kwargs})
        inputs = Inputs(input_features=f"feature-{len(self.calls)}")
        if kwargs["is_first_audio_chunk"]:
            inputs.update(input_ids=[[1, 2]], num_delay_tokens=self.num_delay_tokens)
        return inputs

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert kwargs == {"skip_special_tokens": True}
        return " ".join(self.tokenizer.pieces[token] for token in token_ids)


class Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> object:
        features = list(kwargs["input_features"])
        self.calls.append({**kwargs, "input_features": features})
        return type("Output", (), {"sequences": [[1, 2, 99, 10, 99, 11, 99]]})()


def audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(12, dtype="<i2").tobytes())
    return NormalizedAudio(path, AudioMetadata(path.name, 1.0))


def test_voxtral_uses_one_continuous_generation_and_end_only_word_timestamps(
    tmp_path: Path,
) -> None:
    processor = Processor()
    processor.num_right_pad_tokens = lambda: 1  # type: ignore[method-assign]
    model = Model()
    transcriber = VoxtralTranscriber("/models/voxtral", "cpu")
    transcriber._processor = processor
    transcriber._model = model

    words = transcriber.transcribe(audio(tmp_path))

    assert [(word.text, word.start, word.end) for word in words] == [
        ("Hallo", None, 0.00025),
        ("Welt", None, 0.00075),
    ]
    assert len(model.calls) == 1
    assert model.calls[0]["input_features"] == ["feature-1", "feature-2", "feature-3", "feature-4"]
    assert [call["is_first_audio_chunk"] for call in processor.calls] == [True, False, False, False]
    assert all(call["is_streaming"] is True for call in processor.calls)
    assert transcriber.backend_metrics["stream_buffers_processed"] == 4.0
    assert transcriber.backend_metrics["native_emission_groups"] == 2.0
    assert transcriber.backend_metrics["multi_word_emission_groups"] == 0.0
    assert transcriber.backend_metrics["inferred_final_emission_groups"] == 0.0
    assert transcriber.backend_configuration["num_right_pad_tokens"] == 1


def test_voxtral_timestamp_parser_rejects_text_without_a_native_end_marker() -> None:
    with pytest.raises(VoxtralTimestampError, match=r"without a \[STREAMING_WORD\]"):
        parse_voxtral_words([10], ["Hallo"], lambda _: "Hallo", 1, 0.08, 1.0)


def test_voxtral_timestamp_parser_infers_a_final_group_after_the_last_native_marker() -> None:
    metrics: dict[str, float] = {}
    words = parse_voxtral_words(
        [99, 10],
        ["[STREAMING_WORD]", "Hallo"],
        lambda _: "Hallo",
        6,
        0.08,
        1.0,
        metrics,
    )
    assert [(word.text, word.start, word.end) for word in words] == [("Hallo", None, 1.0)]
    assert metrics == {
        "inferred_final_emission_groups": 1.0,
        "inferred_final_words": 1.0,
    }


def test_voxtral_timestamp_parser_distributes_a_multi_word_final_tail() -> None:
    metrics: dict[str, float] = {}
    words = parse_voxtral_words(
        [99, 10],
        ["[STREAMING_WORD]", "bis"],
        lambda _: "bis morgen dann",
        0,
        0.08,
        1.0,
        metrics,
    )
    assert [(word.text, word.end) for word in words] == [
        ("bis", 1 / 3),
        ("morgen", 2 / 3),
        ("dann", 1.0),
    ]
    assert metrics == {
        "inferred_final_emission_groups": 1.0,
        "inferred_final_words": 3.0,
        "multi_word_emission_groups": 1.0,
    }


def test_voxtral_timestamp_parser_keeps_marker_closed_groups_and_clamps_them() -> None:
    words = parse_voxtral_words(
        [10, 99, 11, 99],
        ["Hallo", "[STREAMING_WORD]", "Welt", "[STREAMING_WORD]"],
        lambda ids: "Hallo" if ids == [10] else "Welt",
        0,
        0.8,
        1.0,
    )
    assert [(word.text, word.end) for word in words] == [("Hallo", 0.8), ("Welt", 1.0)]


def test_voxtral_timestamp_parser_ignores_special_tokens_after_a_marker() -> None:
    assert parse_voxtral_words(
        [99, 2],
        ["[STREAMING_WORD]", "</s>"],
        lambda _: "",
        0,
        0.08,
        1.0,
    ) == []
