import wave
from pathlib import Path

import numpy as np

from meeting_transcriber.models import ASRWord, AudioMetadata, AudioSegment, NormalizedAudio
from meeting_transcriber.transcription.qwen.transcriber import QwenTranscriber


class Recognizer:
    model_reference = "/models/qwen"
    device = "cpu"
    dtype_name = "float32"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load(self) -> None:
        self.events.append("recognizer-load")

    def recognize(self, segment: AudioSegment) -> str:
        self.events.append(f"recognize-{segment.index}")
        return "eins zwei" if segment.index == 0 else ""

    def release(self) -> None:
        self.events.append("recognizer-release")


class Aligner:
    model_reference = "/models/aligner"
    max_segment_duration = 300.0
    alignment_metrics = {"interpolated_word_timestamps": 2.0}

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load(self) -> None:
        self.events.append("aligner-load")

    def align(self, segment: AudioSegment, transcript: str) -> list[ASRWord]:
        self.events.append(f"align-{segment.index}-{transcript}")
        return [ASRWord("eins", 0.1, start=0.0)]

    def release(self) -> None:
        self.events.append("aligner-release")


def audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "meeting.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(48_000, dtype="<i2").tobytes())
    return NormalizedAudio(path, AudioMetadata(path.name, 3.0))


def test_qwen_runs_recognition_then_alignment_with_one_load_per_model(tmp_path: Path) -> None:
    events: list[str] = []
    transcriber = QwenTranscriber(Recognizer(events), Aligner(events), 2, 1)  # type: ignore[arg-type]

    transcriber.load()
    words = transcriber.transcribe(audio(tmp_path))

    assert events == [
        "recognizer-load",
        "recognize-0",
        "recognize-1",
        "recognizer-release",
        "aligner-load",
        "align-0-eins zwei",
        "aligner-release",
    ]
    assert [word.text for word in words] == ["eins"]
    assert transcriber.backend_models == {"forced_aligner_model": "/models/aligner"}
    assert transcriber.backend_metrics["interpolated_word_timestamps"] == 2.0
