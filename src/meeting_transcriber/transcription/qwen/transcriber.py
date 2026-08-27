"""Two-phase Qwen adapter hidden behind the common ASR interface."""

from __future__ import annotations

import time
from dataclasses import dataclass

from meeting_transcriber.audio.segmenter import AudioSegmenter
from meeting_transcriber.models import ASRWord, AudioSegment, NormalizedAudio
from meeting_transcriber.runtime.lifecycle import release_model
from meeting_transcriber.transcription.base import TranscriberCapabilities
from meeting_transcriber.transcription.qwen.forced_aligner import QwenForcedAligner
from meeting_transcriber.transcription.qwen.recognizer import QwenRecognizer
from meeting_transcriber.transcription.segments import reconcile_segment_words


@dataclass(frozen=True)
class RecognizedSegment:
    """Qwen ASR text retained until the forced-alignment phase completes."""

    segment: AudioSegment
    text: str


class QwenTranscriber:
    """Recognize internally bounded segments, then align and reconcile them."""

    capabilities = TranscriberCapabilities(True, True, False, False, requires_forced_alignment=True)

    def __init__(
        self,
        recognizer: QwenRecognizer,
        aligner: QwenForcedAligner,
        segment_duration: float = 240.0,
        segment_overlap: float = 15.0,
    ) -> None:
        self._recognizer = recognizer
        self._aligner = aligner
        self.model_reference = recognizer.model_reference
        self.device = recognizer.device
        self.dtype_name = recognizer.dtype_name
        self.backend_metrics: dict[str, float] = {}
        self.backend_models = {"qwen_aligner_model": aligner.model_reference}
        self.recognized_segments: list[RecognizedSegment] = []
        self._recognizer_loaded = False
        self._segmenter = AudioSegmenter(segment_duration, segment_overlap)
        self.backend_configuration = {
            "segment_duration_seconds": segment_duration,
            "segment_overlap_seconds": segment_overlap,
            "forced_aligner_max_segment_seconds": 300.0,
        }

    def load(self) -> None:
        """Load Qwen ASR once before recognition begins."""
        if self._recognizer_loaded:
            return
        started = time.monotonic()
        self._recognizer.load()
        self._recognizer_loaded = True
        self.backend_metrics["qwen_asr_model_load_seconds"] = time.monotonic() - started

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Run Qwen's complete two-phase lifecycle for one whole meeting."""
        segments = self._segmenter.segment(audio)
        self.backend_metrics["segments_processed"] = float(len(segments))
        self.load()
        recognized = self._recognize_segments(segments)
        self.recognized_segments = recognized
        if not recognized:
            return []
        return reconcile_segment_words(segments, self._align_segments(segments, recognized))

    def _recognize_segments(self, segments: list[AudioSegment]) -> list[RecognizedSegment]:
        started = time.monotonic()
        try:
            recognized = [
                RecognizedSegment(segment, text)
                for segment in segments
                if (text := self._recognizer.recognize(segment))
            ]
            self.backend_metrics["qwen_asr_inference_seconds"] = time.monotonic() - started
            return recognized
        finally:
            unload_started = time.monotonic()
            release_model(self._recognizer)
            self._recognizer_loaded = False
            self.backend_metrics["qwen_asr_unload_seconds"] = time.monotonic() - unload_started

    def _align_segments(
        self, segments: list[AudioSegment], recognized: list[RecognizedSegment]
    ) -> dict[int, list[ASRWord]]:
        load_started = time.monotonic()
        self._aligner.load()
        self.backend_metrics["qwen_aligner_model_load_seconds"] = time.monotonic() - load_started
        alignment_started = time.monotonic()
        try:
            words_by_segment: dict[int, list[ASRWord]] = {
                segment.index: [] for segment in segments
            }
            for recognized_segment in recognized:
                words_by_segment[recognized_segment.segment.index] = self._aligner.align(
                    recognized_segment.segment, recognized_segment.text
                )
            self.backend_metrics["qwen_alignment_seconds"] = time.monotonic() - alignment_started
            return words_by_segment
        finally:
            unload_started = time.monotonic()
            release_model(self._aligner)
            self.backend_metrics["qwen_aligner_unload_seconds"] = time.monotonic() - unload_started

    def release(self) -> None:
        """Release either model if a phase exits early."""
        self._recognizer.release()
        self._aligner.release()
        self._recognizer_loaded = False
