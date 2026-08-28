from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_transcriber.errors import QwenAlignmentError
from speech_transcriber.models import AudioSegment
from speech_transcriber.transcription.qwen.forced_aligner import (
    QwenForcedAligner,
    _validate_transcript_coverage,
    normalize_qwen_alignment,
)
from speech_transcriber.transcription.segments import reconcile_segment_words


class Inputs(dict[str, object]):
    def to(self, device: str, dtype: object) -> Inputs:
        return self


class Processor:
    def __init__(self, entries: list[dict[str, object]] | None = None) -> None:
        self.request: dict[str, object] | None = None
        self.entries = entries or [
            {"text": "Guten", "start_time": 0.2, "end_time": 0.6},
            {"text": "Morgen", "start_time": 0.7, "end_time": 1.2},
        ]

    def prepare_forced_aligner_inputs(self, **kwargs: object) -> tuple[Inputs, list[list[str]]]:
        self.request = kwargs
        transcript = kwargs["transcript"]
        assert isinstance(transcript, str)
        return Inputs(input_ids=torch.tensor([[1, 2]])), [transcript.split()]

    def decode_forced_alignment(self, **kwargs: object) -> list[list[dict[str, object]]]:
        assert kwargs["timestamp_token_id"] == 9
        return [self.entries]


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


def test_forced_aligner_reduces_locale_to_base_language_code() -> None:
    aligner = QwenForcedAligner("/models/aligner", "cpu", language="fr-FR")
    processor = Processor()
    aligner._processor = processor
    aligner._model = Model()
    aligner.align(segment(), "Guten Morgen")
    assert processor.request is not None
    assert processor.request["language"] == "fr"


def test_forced_aligner_rejects_material_transcript_divergence() -> None:
    aligner = QwenForcedAligner("/models/aligner", "cpu")
    aligner._processor = Processor()
    aligner._model = Model()
    with pytest.raises(QwenAlignmentError, match="materially diverges"):
        aligner.align(segment(), "eins zwei drei vier fünf sechs sieben acht neun zehn")


def test_normalizer_clips_or_drops_trailing_boundary_overflow() -> None:
    words = normalize_qwen_alignment(
        [
            {"text": "innerhalb", "start_time": 1.8, "end_time": 2.8},
            {"text": "danach", "start_time": 2.1, "end_time": 2.5},
        ],
        segment(),
    )
    assert [(word.text, word.start, word.end) for word in words] == [("innerhalb", 1.8, 2.0)]


def test_forced_aligner_records_boundary_overflow_metrics() -> None:
    aligner = QwenForcedAligner("/models/aligner", "cpu")
    aligner._processor = Processor(
        [
            {"text": "innerhalb", "start_time": 1.8, "end_time": 2.8},
            {"text": "danach", "start_time": 2.1, "end_time": 2.5},
        ]
    )
    aligner._model = Model()
    aligner.reset_alignment_metrics()

    assert [word.text for word in aligner.align(segment(), "innerhalb danach")] == ["innerhalb"]
    assert aligner.alignment_metrics["interpolated_word_timestamps"] == 0.0
    assert aligner.alignment_metrics["boundary_overflow_words_clipped"] == 1.0
    assert aligner.alignment_metrics["boundary_overflow_words_dropped"] == 1.0
    assert aligner.alignment_metrics["max_boundary_overflow_seconds"] == pytest.approx(0.8)


def test_normalizer_rejects_overflow_that_begins_away_from_the_boundary() -> None:
    with pytest.raises(QwenAlignmentError, match="beyond the audio chunk"):
        normalize_qwen_alignment(
            [{"text": "bad", "start_time": 0.0, "end_time": 2.8}], segment()
        )


def test_normalizer_rejects_alignment_with_only_outside_words() -> None:
    with pytest.raises(QwenAlignmentError, match="no in-window words"):
        normalize_qwen_alignment(
            [{"text": "outside", "start_time": 2.0, "end_time": 2.5}], segment()
        )


def test_forced_aligner_rejects_materially_dropped_trailing_suffix() -> None:
    aligner = QwenForcedAligner("/models/aligner", "cpu")
    aligner._processor = Processor(
        [{"text": "first", "start_time": 1.0, "end_time": 1.5}]
        + [
            {"text": f"outside-{index}", "start_time": 2.0, "end_time": 2.5}
            for index in range(9)
        ]
    )
    aligner._model = Model()
    with pytest.raises(QwenAlignmentError, match="materially diverges"):
        aligner.align(segment(), "eins zwei drei vier fünf sechs sieben acht neun zehn")


def test_coverage_uses_aligner_tokens_for_a_no_space_language() -> None:
    expected = ["你", "好", "世", "界"]
    words = [
        normalize_qwen_alignment(
            [{"text": token, "start_time": index * 0.2, "end_time": (index + 1) * 0.2}],
            segment(),
        )[0]
        for index, token in enumerate(expected)
    ]
    _validate_transcript_coverage(expected, words)


def test_qwen_reconciler_converts_segment_boundaries_to_recording_offsets() -> None:
    words = normalize_qwen_alignment(
        [{"text": "Wort", "start_time": 0.2, "end_time": 0.6}], segment()
    )
    merged = reconcile_segment_words([segment()], {3: words})
    assert [(word.start, word.end) for word in merged] == [(12.2, 12.6)]


def test_normalizer_interpolates_collapsed_word_timestamp_runs() -> None:
    words = normalize_qwen_alignment(
        [
            {"text": "eins", "start_time": 0.0, "end_time": 0.4},
            {"text": "zwei", "start_time": 0.5, "end_time": 0.5},
            {"text": "drei", "start_time": 0.5, "end_time": 0.5},
            {"text": "vier", "start_time": 0.6, "end_time": 0.9},
        ],
        segment(),
    )

    assert [(word.text, word.start, word.end) for word in words] == [
        ("eins", 0.0, 0.4),
        ("zwei", 0.4, 0.5),
        ("drei", 0.5, 0.6),
        ("vier", 0.6, 0.9),
    ]


def test_normalizer_repairs_collapsed_word_without_anchor_space() -> None:
    words = normalize_qwen_alignment(
        [
            {"text": "eins", "start_time": 0.0, "end_time": 0.5},
            {"text": "zwei", "start_time": 0.5, "end_time": 0.5},
            {"text": "drei", "start_time": 0.5, "end_time": 0.8},
        ],
        segment(),
    )

    assert [(word.text, word.start, word.end) for word in words] == [
        ("eins", 0.0, 0.5),
        ("zwei", 0.46, 0.54),
        ("drei", 0.5, 0.8),
    ]


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"text": "", "start_time": 0, "end_time": 1}],
        [{"text": "bad", "start_time": 2, "end_time": 1}],
        [{"text": "bad", "start_time": float("nan"), "end_time": 1}],
        [{"text": "bad", "start_time": 0, "end_time": 3}],
        [
            {"text": "first", "start_time": 0.5, "end_time": 1.0},
            {"text": "second", "start_time": 0.2, "end_time": 0.8},
        ],
    ],
)
def test_rejects_malformed_or_out_of_range_alignment(entries: object) -> None:
    with pytest.raises(QwenAlignmentError):
        normalize_qwen_alignment(entries, segment())
