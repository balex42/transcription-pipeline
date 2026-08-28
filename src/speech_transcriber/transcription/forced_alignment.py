"""Reusable recognition and forced-alignment orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from speech_transcriber.audio.segmenter import AudioSegmenter
from speech_transcriber.models import ASRWord, AudioSegment, NormalizedAudio
from speech_transcriber.runtime.lifecycle import release_model
from speech_transcriber.transcription.base import TranscriberCapabilities
from speech_transcriber.transcription.segments import reconcile_segment_words


class SegmentRecognizer(Protocol):
    """Recognize text from one bounded audio segment."""

    model_reference: str
    device: str
    dtype_name: str

    def load(self) -> None:
        """Load recognition resources."""

    def recognize(self, segment: AudioSegment) -> str:
        """Return transcript text for one segment."""

    def release(self) -> None:
        """Release recognition resources."""


class ForcedAligner(Protocol):
    """Generate word timings for a recognized audio segment."""

    model_reference: str
    max_segment_duration: float

    def load(self) -> None:
        """Load alignment resources."""

    def align(self, segment: AudioSegment, transcript: str) -> list[ASRWord]:
        """Return segment-relative word timings for transcript text."""

    def release(self) -> None:
        """Release alignment resources."""


@dataclass(frozen=True)
class RecognizedSegment:
    """Text retained until a separately loaded aligner creates word timings."""

    segment: AudioSegment
    text: str


class ForcedAlignmentTranscriber:
    """Recognize all segments, then align them after releasing the recognizer."""

    def __init__(
        self,
        recognizer: SegmentRecognizer,
        aligner: ForcedAligner,
        capabilities: TranscriberCapabilities,
        segment_duration: float,
        segment_overlap: float,
    ) -> None:
        self._recognizer = recognizer
        self._aligner = aligner
        self.capabilities = capabilities
        self.model_reference = recognizer.model_reference
        self.device = recognizer.device
        self.dtype_name = recognizer.dtype_name
        self.backend_metrics: dict[str, float] = {}
        self.backend_models = {"forced_aligner_model": aligner.model_reference}
        self._recognizer_loaded = False
        self._segmenter = AudioSegmenter(segment_duration, segment_overlap)
        self.backend_configuration: dict[str, object] = {
            "segment_duration_seconds": segment_duration,
            "segment_overlap_seconds": segment_overlap,
            "forced_aligner_max_segment_seconds": aligner.max_segment_duration,
        }

    def load(self) -> None:
        """Load recognition resources before the recognition phase."""
        if self._recognizer_loaded:
            return
        self._recognizer.load()
        self._recognizer_loaded = True

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Run sequential recognition and alignment phases for one recording."""
        segments = self._segmenter.segment(audio)
        self.backend_metrics["segments_processed"] = float(len(segments))
        self.load()
        recognized = self._recognize_segments(segments)
        if not recognized:
            return []
        return reconcile_segment_words(segments, self._align_segments(segments, recognized))

    def _recognize_segments(self, segments: list[AudioSegment]) -> list[RecognizedSegment]:
        started = time.monotonic()
        try:
            return [
                RecognizedSegment(segment, text)
                for segment in segments
                if (text := self._recognizer.recognize(segment))
            ]
        finally:
            self.backend_metrics["recognition_seconds"] = time.monotonic() - started
            release_started = time.monotonic()
            release_model(self._recognizer)
            self._recognizer_loaded = False
            self.backend_metrics["recognizer_release_seconds"] = time.monotonic() - release_started

    def _align_segments(
        self, segments: list[AudioSegment], recognized: list[RecognizedSegment]
    ) -> dict[int, list[ASRWord]]:
        load_started = time.monotonic()
        try:
            reset_metrics = getattr(self._aligner, "reset_alignment_metrics", None)
            if callable(reset_metrics):
                reset_metrics()
            self._aligner.load()
            self.backend_metrics["forced_aligner_load_seconds"] = time.monotonic() - load_started
            alignment_started = time.monotonic()
            words_by_segment: dict[int, list[ASRWord]] = {segment.index: [] for segment in segments}
            for recognized_segment in recognized:
                words_by_segment[recognized_segment.segment.index] = self._aligner.align(
                    recognized_segment.segment, recognized_segment.text
                )
            self.backend_metrics["forced_alignment_seconds"] = time.monotonic() - alignment_started
            alignment_metrics = getattr(self._aligner, "alignment_metrics", {})
            if isinstance(alignment_metrics, dict):
                for name, value in alignment_metrics.items():
                    if isinstance(name, str) and isinstance(value, int | float):
                        self.backend_metrics[name] = float(value)
            return words_by_segment
        finally:
            release_started = time.monotonic()
            release_model(self._aligner)
            self.backend_metrics["forced_aligner_release_seconds"] = (
                time.monotonic() - release_started
            )

    def release(self) -> None:
        """Release resources from either phase when the outer pipeline exits."""
        self._recognizer.release()
        self._aligner.release()
        self._recognizer_loaded = False
