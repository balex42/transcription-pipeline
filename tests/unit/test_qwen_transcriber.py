import numpy as np

from meeting_transcriber.models import ASRWord, AudioChunk
from meeting_transcriber.transcription.qwen.transcriber import QwenTranscriber


class Recognizer:
    model_reference = "/models/qwen"
    device = "cpu"
    dtype_name = "float32"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load(self) -> None:
        self.events.append("recognizer-load")

    def recognize(self, chunk: AudioChunk) -> str:
        self.events.append(f"recognize-{chunk.chunk_id}")
        return "eins zwei" if chunk.chunk_id == 0 else ""

    def release(self) -> None:
        self.events.append("recognizer-release")


class Aligner:
    model_reference = "/models/aligner"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load(self) -> None:
        self.events.append("aligner-load")

    def align(self, chunk: AudioChunk, transcript: str) -> list[ASRWord]:
        self.events.append(f"align-{chunk.chunk_id}-{transcript}")
        return [ASRWord("eins", 0.1, start=0.0, chunk_id=chunk.chunk_id)]

    def release(self) -> None:
        self.events.append("aligner-release")


def chunks() -> list[AudioChunk]:
    return [
        AudioChunk(0, 0.0, 2.0, np.zeros(32_000, dtype=np.float32)),
        AudioChunk(1, 1.0, 3.0, np.zeros(32_000, dtype=np.float32)),
    ]


def test_qwen_runs_recognition_then_alignment_with_one_load_per_model() -> None:
    events: list[str] = []
    transcriber = QwenTranscriber(Recognizer(events), Aligner(events))  # type: ignore[arg-type]

    transcriber.load()
    words = transcriber.transcribe_chunks(chunks())

    assert events == [
        "recognizer-load",
        "recognize-0",
        "recognize-1",
        "recognizer-release",
        "aligner-load",
        "align-0-eins zwei",
        "aligner-release",
    ]
    assert [word.text for word in words[0]] == ["eins"]
    assert words[1] == []
    assert transcriber.backend_models == {"qwen_aligner_model": "/models/aligner"}
