from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from speech_transcriber.models import ASRWord, AudioMetadata, AudioSegment, NormalizedAudio
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.forced_alignment import ForcedAlignmentTranscriber


class Recognizer:
    model_reference = "/models/recognizer"
    device = "cpu"
    dtype_name = "float32"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load(self) -> None:
        self.events.append("recognizer-load")

    def recognize(self, segment: AudioSegment) -> str:
        self.events.append(f"recognize-{segment.index}")
        return "eins zwei"

    def release(self) -> None:
        self.events.append("recognizer-release")


class Aligner:
    model_reference = "/models/aligner"
    max_segment_duration = 300.0

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load(self) -> None:
        self.events.append("aligner-load")

    def align(self, segment: AudioSegment, transcript: str) -> list[ASRWord]:
        self.events.append(f"align-{segment.index}-{transcript}")
        return [ASRWord("eins", end=0.4, start=0.1)]

    def release(self) -> None:
        self.events.append("aligner-release")


def audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(48_000, dtype="<i2").tobytes())
    return NormalizedAudio(path, AudioMetadata(path.name, 3.0))


def test_generic_transcriber_releases_recognizer_before_reusing_aligner(tmp_path: Path) -> None:
    events: list[str] = []
    transcriber = ForcedAlignmentTranscriber(
        Recognizer(events),
        Aligner(events),
        TranscriberCapabilities(True, True, False, False, requires_forced_alignment=True),
        2.0,
        1.0,
    )

    words = transcriber.transcribe(audio(tmp_path))

    assert events == [
        "recognizer-load",
        "recognize-0",
        "recognize-1",
        "recognizer-release",
        "aligner-load",
        "align-0-eins zwei",
        "align-1-eins zwei",
        "aligner-release",
    ]
    assert [(word.text, word.start, word.end) for word in words] == [("eins", 0.1, 0.4)]
    assert transcriber.backend_models == {"forced_aligner_model": "/models/aligner"}
