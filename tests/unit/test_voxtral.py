from __future__ import annotations

import builtins
import wave
from pathlib import Path

import numpy as np
import pytest

from speech_transcriber.errors import ModelLoadError, VoxtralTimestampError
from speech_transcriber.models import AudioMetadata, NormalizedAudio
from speech_transcriber.transcription.voxtral.timestamps import parse_voxtral_words
from speech_transcriber.transcription.voxtral.transcriber import (
    VoxtralTranscriber,
    resolve_voxtral_model_path,
)


class Inputs(dict[str, object]):
    def to(self, device: str, dtype: object) -> Inputs:
        self["device"] = device
        self["dtype"] = dtype
        return self


class AudioConfig:
    def __init__(self) -> None:
        self.transcription_delay_ms: float | None = None


class AudioEncoder:
    def __init__(self) -> None:
        self.audio_config = AudioConfig()


class InstructTokenizer:
    def __init__(self) -> None:
        self.audio_encoder = AudioEncoder()


class MistralBackend:
    pass


class Tokenizer:
    pieces = {
        1: "[TRANSCRIBE]",
        2: "[STREAMING_PAD]",
        10: "Hallo",
        11: "Welt",
        99: "[STREAMING_WORD]",
    }

    def __init__(self) -> None:
        self.tokenizer = MistralBackend()
        self.tokenizer.instruct_tokenizer = InstructTokenizer()

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

    class FeatureExtractor:
        hop_length = 4
        win_length = 8

    feature_extractor = FeatureExtractor()

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.tokenizer = Tokenizer()

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
    assert model.calls[0]["do_sample"] is False
    assert [call["is_first_audio_chunk"] for call in processor.calls] == [True, False, False, False]
    assert all(call["is_streaming"] is True for call in processor.calls)
    assert transcriber.backend_metrics["stream_buffers_processed"] == 4.0
    assert transcriber.backend_metrics["native_emission_groups"] == 2.0
    assert transcriber.backend_metrics["multi_word_emission_groups"] == 0.0
    assert transcriber.backend_metrics["inferred_final_emission_groups"] == 0.0
    assert transcriber.backend_configuration["num_right_pad_tokens"] == 1
    assert transcriber.backend_configuration["temperature"] == 0.0
    assert transcriber.backend_configuration["timestamp_offset_tokens"] == 1


def test_voxtral_uses_configured_timestamp_offset(
    tmp_path: Path,
) -> None:
    processor = Processor()
    processor.num_right_pad_tokens = lambda: 1  # type: ignore[method-assign]
    model = Model()
    transcriber = VoxtralTranscriber("/models/voxtral", "cpu", timestamp_offset_tokens=0)
    transcriber._processor = processor
    transcriber._model = model

    words = transcriber.transcribe(audio(tmp_path))

    assert [(word.text, word.end) for word in words] == [
        ("Hallo", 0.0005),
        ("Welt", 0.001),
    ]
    assert transcriber.backend_configuration["timestamp_offset_tokens"] == 0


def test_voxtral_configures_audio_delay_from_request() -> None:
    processor = Processor()
    transcriber = VoxtralTranscriber("/models/voxtral", "cpu", delay_ms=2400)
    transcriber._configure_delay(processor)
    audio_config = transcriber._audio_config(processor)
    assert audio_config.transcription_delay_ms == 2400.0


def test_voxtral_leaves_delay_untouched_when_not_requested() -> None:
    processor = Processor()
    transcriber = VoxtralTranscriber("/models/voxtral", "cpu", delay_ms=None)
    transcriber._configure_delay(processor)
    audio_config = transcriber._audio_config(processor)
    assert audio_config.transcription_delay_ms is None


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


def fake_cache(tmp_path: Path) -> Path:
    """Build a prefetched single-revision HF cache for the Voxtral model."""
    repository = "models--mistralai--Voxtral-Mini-4B-Realtime-2602"
    snapshot = tmp_path / "hub" / repository / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    refs = tmp_path / "hub" / repository / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc123\n", encoding="utf-8")
    return snapshot


def test_load_passes_one_resolved_local_snapshot_to_processor_and_model(
    tmp_path: Path, monkeypatch: object
) -> None:
    snapshot = fake_cache(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # type: ignore[attr-defined]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # type: ignore[attr-defined]
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeProcessor:
            calls.append(("processor", path, kwargs))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeModel:
            calls.append(("model", path, kwargs))
            return cls()

        def to(self, _: str) -> FakeModel:
            return self

        def eval(self) -> FakeModel:
            return self

    import speech_transcriber.transcription.voxtral.transcriber as module

    monkeypatch.setattr(  # type: ignore[attr-defined]
        module,
        "_transformers_voxtral_classes",
        lambda: (FakeProcessor, FakeModel),
    )
    transcriber = VoxtralTranscriber("mistralai/Voxtral-Mini-4B-Realtime-2602", "cpu")
    transcriber.load()

    expected = str(snapshot)
    assert [kind for kind, _, _ in calls] == ["processor", "model"]
    for _, path, kwargs in calls:
        assert path == expected
        assert path.startswith(str(tmp_path))
        assert "models--mistralai--Voxtral-Mini-4B-Realtime-2602" in path
        assert kwargs.get("trust_remote_code") is False
    assert transcriber.backend_models["model_snapshot"] == "abc123"


def test_resolver_prefers_an_existing_absolute_local_directory(tmp_path: Path) -> None:
    local = tmp_path / "voxtral"
    local.mkdir()

    assert resolve_voxtral_model_path(str(local)) == str(local)


def test_resolver_does_not_reinterpret_relative_names(tmp_path: Path) -> None:
    (tmp_path / "voxtral-mini").mkdir()

    with pytest.raises(ModelLoadError, match="offline model cache"):
        resolve_voxtral_model_path("voxtral-mini")


def test_resolver_rejects_missing_absolute_directory(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="does not exist"):
        resolve_voxtral_model_path(str(tmp_path / "missing"))

def test_offline_load_never_contacts_the_hugging_face_hub(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Regression for the real Argo failure: cached snapshot, no Hub access.

    Simulates the production env (``HF_HUB_OFFLINE=1``,
    ``TRANSFORMERS_OFFLINE=1``) and installs an import guard that rejects any
    network-facing hub or transports, then proves that processor and model
    loading happens through the resolved local snapshot path only.
    """
    snapshot = fake_cache(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    calls: list[tuple[str, str, dict[str, object]]] = []

    class LocalOnlyProcessor:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> LocalOnlyProcessor:
            assert Path(path).is_absolute() and Path(path) == snapshot
            assert kwargs.get("trust_remote_code") is False
            calls.append(("processor", path, kwargs))
            return cls()

    class LocalOnlyModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> LocalOnlyModel:
            assert Path(path).is_absolute() and Path(path) == snapshot
            assert kwargs.get("trust_remote_code") is False
            calls.append(("model", path, kwargs))
            return cls()

        def to(self, _: str) -> LocalOnlyModel:
            return self

        def eval(self) -> LocalOnlyModel:
            return self

    real_import = builtins.__import__
    guarded = (
        "huggingface_hub",
        "requests",
        "httpx",
        "urllib.request",
        "http.client",
        "socket",
    )

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in guarded):
            raise ImportError(f"network access blocked in offline test: {name}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    import speech_transcriber.transcription.voxtral.transcriber as module

    monkeypatch.setattr(
        module,
        "_transformers_voxtral_classes",
        lambda: (LocalOnlyProcessor, LocalOnlyModel),
    )
    monkeypatch.setattr(builtins, "__import__", blocked)  # type: ignore[assignment]
    transcriber = VoxtralTranscriber("mistralai/Voxtral-Mini-4B-Realtime-2602", "cpu")
    transcriber.load()

    expected = str(snapshot)
    assert [kind for kind, _, _ in calls] == ["processor", "model"]
    for _, path, _ in calls:
        assert path == expected
    assert transcriber.backend_models["model_snapshot"] == "abc123"
