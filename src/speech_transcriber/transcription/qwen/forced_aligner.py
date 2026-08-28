"""Native Transformers forced-alignment stage for Qwen transcripts."""

from __future__ import annotations

import math
from typing import Any

from speech_transcriber.errors import ModelLoadError, QwenAlignmentError
from speech_transcriber.models import ASRWord, AudioSegment
from speech_transcriber.runtime.device import inference_dtype

_TIMESTAMP_TOLERANCE_SECONDS = 0.25
_TIMESTAMP_GRID_SECONDS = 0.08
_MAX_INTERPOLATED_RUN_SPAN_SECONDS = 0.8
_TRAILING_BOUNDARY_WINDOW_SECONDS = 2.0


class QwenForcedAligner:
    """Generate validated native word boundaries for recognized text."""

    max_segment_duration = 300.0

    def __init__(self, model: str, device: str, language: str = "de-DE") -> None:
        self.model_reference = model
        self.device = device
        self.language = language
        _, self.dtype_name = inference_dtype(device)
        self._model: Any | None = None
        self._processor: Any | None = None
        self.alignment_metrics: dict[str, float] = {}

    def load(self) -> None:
        """Load the native Qwen forced aligner without remote code."""
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoModelForTokenClassification, AutoProcessor

            dtype, _ = inference_dtype(self.device)
            processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_reference, trust_remote_code=False
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_reference,
                dtype=dtype,
                trust_remote_code=False,
            )
            model.to(self.device)
            model.eval()
            self._model, self._processor = model, processor
        except Exception as error:
            raise ModelLoadError(
                f"could not load Qwen forced-aligner model {self.model_reference}: {error}"
            ) from error

    def align(self, segment: AudioSegment, transcript: str) -> list[ASRWord]:
        """Return segment-relative Qwen word intervals for one recognized segment."""
        if not transcript:
            return []
        self.load()
        assert self._model is not None and self._processor is not None
        try:
            import torch

            dtype, _ = inference_dtype(self.device)
            inputs, word_lists = self._processor.prepare_forced_aligner_inputs(
                audio=segment.audio,
                transcript=transcript,
                language=_qwen_language(self.language),
            )
            inputs = inputs.to(self.device, dtype=dtype)
            with torch.inference_mode():
                outputs = self._model(**inputs)
            decoded = self._processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=self._model.config.timestamp_token_id,
            )
            if not isinstance(decoded, list) or len(decoded) != 1:
                raise QwenAlignmentError("Qwen forced aligner returned an unexpected batch shape")
            (
                words,
                repaired_count,
                boundary_metrics,
                interpolation_metrics,
            ) = _normalize_qwen_alignment(decoded[0], segment)
            self.alignment_metrics["interpolated_word_timestamps"] = (
                self.alignment_metrics.get("interpolated_word_timestamps", 0.0) + repaired_count
            )
            self.alignment_metrics["boundary_overflow_words_clipped"] = (
                self.alignment_metrics.get("boundary_overflow_words_clipped", 0.0)
                + boundary_metrics["clipped"]
            )
            self.alignment_metrics["boundary_overflow_words_dropped"] = (
                self.alignment_metrics.get("boundary_overflow_words_dropped", 0.0)
                + boundary_metrics["dropped"]
            )
            self.alignment_metrics["max_boundary_overflow_seconds"] = max(
                self.alignment_metrics.get("max_boundary_overflow_seconds", 0.0),
                boundary_metrics["max_overflow_seconds"],
            )
            self.alignment_metrics["interpolated_timestamp_runs"] = (
                self.alignment_metrics.get("interpolated_timestamp_runs", 0.0)
                + interpolation_metrics["runs"]
            )
            self.alignment_metrics["capped_interpolation_runs"] = (
                self.alignment_metrics.get("capped_interpolation_runs", 0.0)
                + interpolation_metrics["capped_runs"]
            )
            self.alignment_metrics["unrepaired_zero_duration_words"] = (
                self.alignment_metrics.get("unrepaired_zero_duration_words", 0.0)
                + interpolation_metrics["unrepaired_words"]
            )
            self.alignment_metrics["max_interpolation_anchor_gap_seconds"] = max(
                self.alignment_metrics.get("max_interpolation_anchor_gap_seconds", 0.0),
                interpolation_metrics["max_anchor_gap_seconds"],
            )
            _validate_transcript_coverage(word_lists[0], words)
            return words
        except QwenAlignmentError as error:
            raise QwenAlignmentError(
                "Qwen forced alignment failed for "
                f"segment {segment.index} ({segment.start:.3f}-{segment.end:.3f}s) "
                f"with model {self.model_reference}: {error}"
            ) from error
        except Exception as error:
            raise QwenAlignmentError(
                "Qwen forced alignment failed for "
                f"segment {segment.index} ({segment.start:.3f}-{segment.end:.3f}s) "
                f"with model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop forced-aligner model references after the timing pass."""
        self._model = None
        self._processor = None

    def reset_alignment_metrics(self) -> None:
        """Clear per-recording alignment metrics before a new alignment pass."""
        self.alignment_metrics = {
            "interpolated_word_timestamps": 0.0,
            "boundary_overflow_words_clipped": 0.0,
            "boundary_overflow_words_dropped": 0.0,
            "max_boundary_overflow_seconds": 0.0,
            "interpolated_timestamp_runs": 0.0,
            "capped_interpolation_runs": 0.0,
            "unrepaired_zero_duration_words": 0.0,
            "max_interpolation_anchor_gap_seconds": 0.0,
        }


def normalize_qwen_alignment(entries: object, segment: AudioSegment) -> list[ASRWord]:
    """Validate native Qwen word boundaries and normalize them to ``ASRWord``."""
    words, _, _, _ = _normalize_qwen_alignment(entries, segment)
    return words


def _normalize_qwen_alignment(
    entries: object, segment: AudioSegment
) -> tuple[list[ASRWord], int, dict[str, float], dict[str, float]]:
    """Normalize boundaries and repair timestamp-grid collisions."""
    if not isinstance(entries, list) or not entries:
        raise QwenAlignmentError(
            f"Qwen forced aligner returned no words for segment {segment.index}"
        )
    duration = segment.end - segment.start
    previous_start = -_TIMESTAMP_TOLERANCE_SECONDS
    previous_end = -_TIMESTAMP_TOLERANCE_SECONDS
    words: list[ASRWord] = []
    boundary_metrics = {"clipped": 0.0, "dropped": 0.0, "max_overflow_seconds": 0.0}
    for entry in entries:
        if not isinstance(entry, dict):
            raise QwenAlignmentError("Qwen forced aligner returned a non-object word")
        text = entry.get("text")
        start = entry.get("start_time")
        end = entry.get("end_time")
        if not isinstance(text, str) or not text.strip():
            raise QwenAlignmentError("Qwen forced aligner returned an empty word")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise QwenAlignmentError("Qwen forced aligner returned non-numeric word timing")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise QwenAlignmentError("Qwen forced aligner returned non-finite word timing")
        if start < -_TIMESTAMP_TOLERANCE_SECONDS or end < start:
            raise QwenAlignmentError("Qwen forced aligner returned an invalid word interval")
        overflow = max(start, end) - duration
        if start >= duration:
            boundary_metrics["dropped"] += 1
            boundary_metrics["max_overflow_seconds"] = max(
                boundary_metrics["max_overflow_seconds"], overflow
            )
            continue
        if end > duration:
            trailing_window = min(_TRAILING_BOUNDARY_WINDOW_SECONDS, duration / 4)
            if start < duration - trailing_window:
                raise QwenAlignmentError(
                    "Qwen forced aligner returned timing beyond the audio chunk"
                )
            # The classifier timestamp grid can place a trailing word past a hard
            # segment boundary. The overlapping next segment owns its continuation.
            boundary_metrics["clipped"] += 1
            boundary_metrics["max_overflow_seconds"] = max(
                boundary_metrics["max_overflow_seconds"], overflow
            )
            end = duration
        if start + _TIMESTAMP_TOLERANCE_SECONDS < previous_start or end < previous_end:
            raise QwenAlignmentError("Qwen forced aligner returned non-monotonic word timing")
        words.append(
            ASRWord(
                text=text.strip(),
                start=max(start, 0.0),
                end=min(end, duration),
            )
        )
        previous_start, previous_end = start, end
    if not words:
        raise QwenAlignmentError(
            f"Qwen forced aligner returned no in-window words for segment {segment.index}"
        )
    repaired, repaired_count, interpolation_metrics = _repair_zero_duration_words(words, duration)
    return repaired, repaired_count, boundary_metrics, interpolation_metrics


def _repair_zero_duration_words(
    words: list[ASRWord], duration: float
) -> tuple[list[ASRWord], int, dict[str, float]]:
    """Localize words collapsed onto one Qwen timestamp-grid position.

    Qwen's 80 ms grid can assign the same start/end position to several words.
    Do not spread them through a long gap to the next acoustic anchor: missing
    transcript text would turn that fabricated span into wrong speaker labels.
    """
    repaired = list(words)
    repaired_count = 0
    interpolation_metrics = {
        "runs": 0.0,
        "capped_runs": 0.0,
        "unrepaired_words": 0.0,
        "max_anchor_gap_seconds": 0.0,
    }
    index = 0
    while index < len(repaired):
        word = repaired[index]
        if word.start is None or word.end != word.start:
            index += 1
            continue
        center = word.start
        run_end = index + 1
        while run_end < len(repaired):
            candidate = repaired[run_end]
            if (
                candidate.start is None
                or candidate.end != candidate.start
                or candidate.start != center
            ):
                break
            run_end += 1
        count = run_end - index
        left = repaired[index - 1].end if index else 0.0
        right = repaired[run_end].start if run_end < len(repaired) else duration
        assert left is not None and right is not None
        available_span = max(right - left, 0.0)
        natural_span = min(
            _TIMESTAMP_GRID_SECONDS * count,
            _MAX_INTERPOLATED_RUN_SPAN_SECONDS,
        )
        interpolation_metrics["runs"] += 1
        interpolation_metrics["max_anchor_gap_seconds"] = max(
            interpolation_metrics["max_anchor_gap_seconds"], available_span
        )
        if available_span <= 0.0:
            interpolation_metrics["unrepaired_words"] += count
            index = run_end
            continue
        span = min(natural_span, available_span)
        if span < available_span:
            interpolation_metrics["capped_runs"] += 1
        start = min(max(center - span / 2, left), right - span)
        end = start + span
        interval = (end - start) / count
        for offset in range(count):
            original = repaired[index + offset]
            repaired[index + offset] = ASRWord(
                text=original.text,
                start=start + interval * offset,
                end=start + interval * (offset + 1),
                confidence=original.confidence,
            )
        repaired_count += count
        index = run_end
    return repaired, repaired_count, interpolation_metrics


def _validate_transcript_coverage(expected_words: list[str], words: list[ASRWord]) -> None:
    """Reject alignments that omit a material portion of the recognized text."""
    expected_count = len(expected_words)
    allowed_difference = max(2, round(expected_count * 0.2))
    if abs(expected_count - len(words)) > allowed_difference:
        raise QwenAlignmentError(
            "Qwen forced aligner word count materially diverges from the recognized transcript"
        )


def _qwen_language(language: str) -> str:
    """Reduce a locale like ``de-DE`` to the base code Qwen expects (``de``)."""
    return language.split("-", 1)[0]
