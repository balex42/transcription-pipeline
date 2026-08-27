from __future__ import annotations

import numpy as np
import pytest
import torch

from meeting_transcriber.errors import QwenAlignmentError
from meeting_transcriber.models import AudioSegment
from meeting_transcriber.transcription.qwen.forced_aligner import (
    QwenForcedAligner,
    normalize_qwen_alignment,
)
from meeting_transcriber.transcription.segments import reconcile_segment_words


class Inputs(dict[str, object]):
    def to(self, device: str, dtype: object) -> Inputs:
        return self


class Processor:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def prepare_forced_aligner_inputs(self, **kwargs: object) -> tuple[Inputs, list[list[str]]]:
        self.request = kwargs
        return Inputs(input_ids=torch.tensor([[1, 2]])), [["Guten", "Morgen"]]

    def decode_forced_alignment(self, **kwargs: object) -> list[list[dict[str, object]]]:
        assert kwargs["timestamp_token_id"] == 9
        return [[
            {"text": "Guten", "start_time": 0.2, "end_time": 0.6},
            {"text": "Morgen", "start_time": 0.7, "end_time": 1.2},
        ]]


class Model:
    class Config:
        timestamp_token_id = 9

    config = Config()

    def __call__(self, **kwargs: object) -> object:
        return type("Output", (), {"logits": torch.tensor([1.0])})()


def segment() -> AudioSegment:
    return AudioSegment(3, 12.0, 14.0, np.zeros(32_000, dtype=np.float32))


def test_forced_aligner_normalizes_native_word_boundaries() -> None:
    aligner = QwenForcedAligner("/models/aligner", "cpu")
    processor = Processor()
    aligner._processor = processor
    aligner._model = Model()

    words = aligner.align(segment(), "Guten Morgen")
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Guten", 0.2, 0.6),
        ("Morgen", 0.7, 1.2),
    ]
    assert processor.request is not None
    assert processor.request["language"] == "de"


def test_forced_aligner_rejects_material_transcript_divergence() -> None:
    aligner = QwenForcedAligner("/models/aligner", "cpu")
    aligner._processor = Processor()
    aligner._model = Model()
    with pytest.raises(QwenAlignmentError, match="materially diverges"):
        aligner.align(segment(), "eins zwei drei vier fünf sechs sieben acht neun zehn")


def test_qwen_reconciler_converts_segment_boundaries_to_meeting_offsets() -> None:
    words = normalize_qwen_alignment(
        [{"text": "Wort", "start_time": 0.2, "end_time": 0.6}], segment()
    )
    merged = reconcile_segment_words([segment()], {3: words})
    assert [(word.start, word.end) for word in merged] == [(12.2, 12.6)]


def test_normalizer_clips_or_drops_trailing_boundary_overflow() -> None:
    words = normalize_qwen_alignment(
        [
            {"text": "innerhalb", "start_time": 1.8, "end_time": 2.8},
            {"text": "danach", "start_time": 2.1, "end_time": 2.5},
        ],
        segment(),
    )

    assert [(word.text, word.start, word.end) for word in words] == [("innerhalb", 1.8, 2.0)]


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"text": "", "start_time": 0, "end_time": 1}],
        [{"text": "bad", "start_time": 2, "end_time": 1}],
        [{"text": "bad", "start_time": float("nan"), "end_time": 1}],
        [
            {"text": "first", "start_time": 0.5, "end_time": 1.0},
            {"text": "second", "start_time": 0.2, "end_time": 0.8},
        ],
    ],
)
def test_rejects_malformed_or_out_of_range_alignment(entries: object) -> None:
    with pytest.raises(QwenAlignmentError):
        normalize_qwen_alignment(entries, segment())
