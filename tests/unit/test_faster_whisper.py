from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from speech_transcriber.errors import ASROutputError
from speech_transcriber.models import AudioMetadata, NormalizedAudio
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.faster_whisper import (
    FasterWhisperTranscriber,
    flatten_segment_words,
    resolve_model_path,
    whisper_language,
)


class Word:
    def __init__(self, word: str, start: float, end: float, probability: float) -> None:
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class Segment:
    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        words: list[Word] | None,
        language: str = "de",
        language_probability: float = 0.98,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.words = words
        self.language = language
        self.language_probability = language_probability


class Model:
    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio: str | Path, **kwargs: object) -> list[Segment]:
        self.calls.append({"audio": audio, **kwargs})
        return self.segments


def audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(16_000, dtype="<i2").tobytes())
    return NormalizedAudio(path, AudioMetadata(path.name, 1.0))


def test_capabilities_declare_native_word_timestamps_and_no_forced_alignment() -> None:
    transcriber = FasterWhisperTranscriber("/models/faster-whisper", "cpu")
    assert transcriber.capabilities == TranscriberCapabilities(True, True, True, True)
    assert transcriber.capabilities.requires_forced_alignment is False
    assert transcriber.capabilities.streaming is False


def test_transcribe_maps_words_timestamps_and_confidence(tmp_path: Path) -> None:
    model = Model(
        [
            Segment(
                0.0,
                1.0,
                "Hallo Welt.",
                [
                    Word("Hallo", 0.0, 0.4, 0.99),
                    Word(" Welt", 0.4, 0.8, 0.95),
                    Word(".", 0.8, 0.9, 0.9),
                ],
            )
        ]
    )
    transcriber = FasterWhisperTranscriber("/models/faster-whisper", "cpu", "de-DE")
    transcriber._model = model

    words = transcriber.transcribe(audio(tmp_path))

    assert [(word.text, word.start, word.end, word.confidence) for word in words] == [
        ("Hallo", 0.0, 0.4, 0.99),
        ("Welt", 0.4, 0.8, 0.95),
        (".", 0.8, 0.9, 0.9),
    ]
    assert model.calls[0]["word_timestamps"] is True
    assert model.calls[0]["vad_filter"] is False
    assert model.calls[0]["language"] == "de"
    assert model.calls[0]["beam_size"] is None
    assert transcriber.backend_configuration["vad_filter"] is False
    assert transcriber.backend_configuration["word_timestamps"] is True
    assert transcriber.backend_metrics["detected_language_probability"] == 0.98
    assert transcriber.backend_configuration["detected_language"] == "de"


def test_transcribe_preserves_punctuation_and_whitespace_boundaries(tmp_path: Path) -> None:
    model = Model(
        [
            Segment(
                0.0,
                1.0,
                "Guten Morgen!",
                [Word(" Guten", 0.0, 0.4, 0.9), Word(" Morgen!", 0.4, 1.0, 0.9)],
            )
        ]
    )
    transcriber = FasterWhisperTranscriber("/models/faster-whisper", "cpu")
    transcriber._model = model

    words = transcriber.transcribe(audio(tmp_path))

    assert [word.text for word in words] == ["Guten", "Morgen!"]
    assert [word.start for word in words] == [0.0, 0.4]
    assert [word.end for word in words] == [0.4, 1.0]


def test_transcribe_skips_segments_without_word_timestamps(tmp_path: Path) -> None:
    model = Model([Segment(0.0, 1.0, "Hallo", None)])
    transcriber = FasterWhisperTranscriber("/models/faster-whisper", "cpu")
    transcriber._model = model

    assert transcriber.transcribe(audio(tmp_path)) == []


def test_transcribe_rejects_words_without_numeric_timestamps(tmp_path: Path) -> None:
    segment = Segment(0.0, 1.0, "Hallo", [Word("Hallo", 0.0, 0.5, 0.9)])
    segment.words[0].start = None  # type: ignore[assignment]
    transcriber = FasterWhisperTranscriber("/models/faster-whisper", "cpu")
    transcriber._model = Model([segment])

    with pytest.raises(ASROutputError, match="missing numeric start/end"):
        transcriber.transcribe(audio(tmp_path))


def test_transcribe_rejects_words_ending_before_they_start(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber("/models/faster-whisper", "cpu")
    transcriber._model = Model([Segment(0.0, 1.0, "Hallo", [Word("Hallo", 0.5, 0.4, 0.9)])])

    with pytest.raises(ASROutputError, match="ends before it starts"):
        transcriber.transcribe(audio(tmp_path))


def test_transcribe_without_language_permits_detection(tmp_path: Path) -> None:
    model = Model([Segment(0.0, 1.0, "Hallo", [Word("Hallo", 0.0, 0.5, 0.9)])])
    transcriber = FasterWhisperTranscriber("/models/faster-whisper", "cpu", language=None)
    transcriber._model = model

    transcriber.transcribe(audio(tmp_path))

    assert model.calls[0]["language"] is None
    assert transcriber.backend_configuration["language"] is None


def test_load_resolves_cached_snapshot_and_records_provenance(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "models--Systran--faster-whisper-large-v3" / "snapshots" / "main"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    class FakeWhisperModel:
        def __init__(self, model_path: str, device: str, compute_type: str) -> None:
            self.model_path = model_path
            self.kwargs = {"device": device, "compute_type": compute_type}

    import speech_transcriber.transcription.faster_whisper as module

    monkeypatch.setattr(module, "_create_whisper_model", FakeWhisperModel)  # type: ignore[attr-defined]
    transcriber = FasterWhisperTranscriber("Systran/faster-whisper-large-v3", "cuda")
    transcriber.load()

    assert transcriber._model is not None
    assert transcriber._model.model_path == str(snapshot)
    assert transcriber._model.kwargs == {"device": "cuda", "compute_type": "float16"}
    assert transcriber.backend_models["model_path"] == str(snapshot)
    assert transcriber.runtime_provenance.name == "faster-whisper"
    assert "ctranslate2" in transcriber.runtime_provenance.components
    assert "huggingface_hub" in transcriber.runtime_provenance.components


def test_load_uses_absolute_model_path_directly(tmp_path: Path, monkeypatch: object) -> None:
    local = tmp_path / "model"
    local.mkdir()

    class FakeWhisperModel:
        def __init__(self, model_path: str, device: str, compute_type: str) -> None:
            self.model_path = model_path

    import speech_transcriber.transcription.faster_whisper as module

    monkeypatch.setattr(module, "_create_whisper_model", FakeWhisperModel)  # type: ignore[attr-defined]
    transcriber = FasterWhisperTranscriber(str(local), "cpu")
    transcriber.load()

    assert transcriber._model is not None
    assert transcriber._model.model_path == str(local)


def test_resolve_model_path_uses_cached_snapshot_when_present(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "models--Systran--faster-whisper-large-v3" / "snapshots" / "main"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(cache))

    assert (
        resolve_model_path("Systran/faster-whisper-large-v3") == str(snapshot)
    )


def test_resolve_model_path_uses_single_revision_snapshot(
    tmp_path: Path, monkeypatch: object
) -> None:
    cache = tmp_path / "cache"
    revision = (
        cache / "hub" / "models--Systran--faster-whisper-large-v3" / "snapshots" / "abc123"
    )
    revision.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(cache))

    assert (
        resolve_model_path("Systran/faster-whisper-large-v3") == str(revision)
    )


def test_resolve_model_path_falls_back_to_repository_id(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    assert (
        resolve_model_path("Systran/faster-whisper-large-v3")
        == "Systran/faster-whisper-large-v3"
    )


def test_resolve_model_path_ignores_hf_home_without_snapshot(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    assert (
        resolve_model_path("Systran/faster-whisper-large-v3")
        == "Systran/faster-whisper-large-v3"
    )


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("de-DE", "de"),
        ("de", "de"),
        ("en-US", "en"),
        (None, None),
    ],
)
def test_whisper_language_normalization(language: str | None, expected: str | None) -> None:
    assert whisper_language(language) == expected


def test_flatten_segment_words_handles_generator_segments() -> None:
    def segments() -> object:
        yield Segment(0.0, 1.0, "Hallo", [Word("Hallo", 0.0, 0.5, 0.9)])

    words = flatten_segment_words(segments())
    assert [(word.text, word.start, word.end, word.confidence) for word in words] == [
        ("Hallo", 0.0, 0.5, 0.9)
    ]


def test_flatten_segment_words_ignores_empty_text() -> None:
    assert flatten_segment_words(
        [Segment(0.0, 1.0, "", [Word("", 0.0, 0.5, 0.9), Word("  ", 0.5, 1.0, 0.9)])]
    ) == []
