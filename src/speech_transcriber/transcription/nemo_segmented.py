"""Shared segmented NeMo recognition for FastConformer TDT checkpoints."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import cast

from speech_transcriber.audio.segmenter import AudioSegmenter
from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.models import ASRWord, AudioSegment, NormalizedAudio
from speech_transcriber.transcription import nemo_support
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities
from speech_transcriber.transcription.segments import reconcile_segment_words

LOGGER = logging.getLogger(__name__)

TIMESTAMP_TOLERANCE_SECONDS = 1e-3


class SegmentedNeMoTranscriber(Transcriber):
    """Bound NeMo FastConformer inference to overlapping PCM segments."""

    backend_name: str
    checkpoint_file: str
    capabilities = TranscriberCapabilities(True, True, True, True)

    def __init__(
        self,
        model: str,
        device: str,
        segment_duration: float = 180.0,
        segment_overlap: float = 15.0,
    ) -> None:
        self.model_reference = model
        self.device = device
        self.dtype_name = "checkpoint-default"
        self._model: object | None = None
        self._segmenter = AudioSegmenter(segment_duration, segment_overlap)
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        self.backend_configuration: dict[str, str | int | float | bool | None] = {
            "segment_duration_seconds": segment_duration,
            "segment_overlap_seconds": segment_overlap,
            "timestamp_mode": "native_word",
            "batch_size": 1,
            "checkpoint_file": self.checkpoint_file,
        }
        self.runtime_provenance = nemo_support.initial_runtime_provenance()

    def load(self) -> None:
        """Load the NeMo checkpoint lazily."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Transcribe overlapping segments and reconcile recording-global words."""
        segments = self._segmenter.segment(audio)
        self.backend_metrics["segments_processed"] = float(len(segments))
        return reconcile_segment_words(
            segments, {segment.index: self._transcribe_segment(segment) for segment in segments}
        )

    def _transcribe_segment(self, segment: AudioSegment) -> list[ASRWord]:
        model = self._load()
        words = flatten_nemo_words(
            self._transcribe(model, segment), subject=self.backend_name, scope="segment"
        )
        return validate_nemo_segment_words(words, segment, subject=self.backend_name)

    def _transcribe(self, model: object, segment: AudioSegment) -> Sequence[object]:
        try:
            return cast(
                Sequence[object],
                model.transcribe(  # type: ignore[attr-defined]
                    [segment.audio], batch_size=1, return_hypotheses=True, timestamps=True
                ),
            )
        except Exception as error:
            raise ASROutputError(
                f"{self.backend_name} recognition failed for segment {segment.index}: {error}"
            ) from error

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        checkpoint, snapshot = nemo_support.resolve_checkpoint_path(
            self.model_reference, self.checkpoint_file, subject=f"{self.backend_name} model"
        )
        try:
            self._model = self._restore_model(checkpoint)
        except Exception as error:
            raise ModelLoadError(
                f"could not load {self.backend_name} model {self.model_reference}: {error}"
            ) from error
        self.backend_models["model_file"] = self.checkpoint_file
        if snapshot:
            self.backend_models["model_snapshot"] = snapshot
        LOGGER.info(
            "loading %s from snapshot", self.backend_name, extra={"snapshot": snapshot or "direct"}
        )
        self.runtime_provenance = nemo_support.nemo_runtime_provenance()
        return self._model

    def _restore_model(self, checkpoint: str) -> object:
        raise NotImplementedError

    def release(self) -> None:
        """Drop model references for sequential GPU execution."""
        self._model = None


def flatten_nemo_words(
    outputs: Sequence[object], *, subject: str, scope: str, duration_seconds: float | None = None
) -> list[ASRWord]:
    """Map native NeMo word timestamps to canonical words for one inference call."""
    if len(outputs) != 1:
        raise ASROutputError(f"{subject} returned {len(outputs)} hypotheses for one {scope}")
    try:
        records = nemo_support.word_timestamp_records(outputs[0], subject=subject)
    except ValueError as error:
        raise ASROutputError(str(error)) from error

    words: list[ASRWord] = []
    previous_end: float | None = None
    for record in records:
        if not isinstance(record, Mapping):
            raise ASROutputError(f"{subject} word timestamp must be an object")
        text = record.get("word")
        start = record.get("start")
        end = record.get("end")
        if not isinstance(text, str) or not text.strip():
            raise ASROutputError(f"{subject} word timestamp is missing text")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ASROutputError(f"{subject} word timestamp is missing numeric start/end values")
        if start < -TIMESTAMP_TOLERANCE_SECONDS or end < start - TIMESTAMP_TOLERANCE_SECONDS:
            raise ASROutputError(f"{subject} word timestamp is invalid: '{text}' {start}-{end}")
        if previous_end is not None and start < previous_end - TIMESTAMP_TOLERANCE_SECONDS:
            raise ASROutputError(
                f"{subject} word timestamps reorder at index {len(words)}: "
                f"'{text}' starts {start} after {previous_end}"
            )
        previous_end = max(float(end), previous_end or float(end))
        confidence_value = record.get("confidence")
        confidence = float(confidence_value) if isinstance(confidence_value, int | float) else None
        words.append(
            ASRWord(text=text.strip(), start=float(start), end=float(end), confidence=confidence)
        )

    if duration_seconds is not None:
        for word in words:
            if word.end > duration_seconds + 1.0:
                raise ASROutputError(
                    f"{subject} word timestamp exceeds recording duration: "
                    f"'{word.text}' ends at {word.end}"
                )
    return words


def validate_nemo_segment_words(
    words: Sequence[ASRWord], segment: AudioSegment, *, subject: str
) -> list[ASRWord]:
    """Reject word bounds outside the segment before global reconciliation."""
    duration = segment.end - segment.start
    for word in words:
        if word.end > duration + 1.0:
            raise ASROutputError(
                f"{subject} word timestamp exceeds the segment duration: "
                f"'{word.text}' ends at {word.end}"
            )
    return list(words)
