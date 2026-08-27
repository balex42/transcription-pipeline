"""Native Transformers forced-alignment stage for Qwen transcripts."""

from __future__ import annotations

import math
import re
from typing import Any

from meeting_transcriber.errors import ModelLoadError, QwenAlignmentError
from meeting_transcriber.models import ASRWord, AudioSegment
from meeting_transcriber.runtime.device import inference_dtype

_TIMESTAMP_TOLERANCE_SECONDS = 0.25


class QwenForcedAligner:
    """Generate validated native word boundaries for recognized German text."""

    max_segment_duration = 300.0

    def __init__(self, model: str, device: str) -> None:
        self.model_reference = model
        self.device = device
        _, self.dtype_name = inference_dtype(device)
        self._model: Any | None = None
        self._processor: Any | None = None

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
                language="de",
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
            words = normalize_qwen_alignment(decoded[0], segment)
            _validate_transcript_coverage(transcript, words)
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


def normalize_qwen_alignment(entries: object, segment: AudioSegment) -> list[ASRWord]:
    """Validate native Qwen word boundaries and normalize them to ``ASRWord``."""
    if not isinstance(entries, list) or not entries:
        raise QwenAlignmentError(
            f"Qwen forced aligner returned no words for segment {segment.index}"
        )
    duration = segment.end - segment.start
    previous_start = -_TIMESTAMP_TOLERANCE_SECONDS
    previous_end = -_TIMESTAMP_TOLERANCE_SECONDS
    words: list[ASRWord] = []
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
        # The classifier timestamp grid can place a trailing word past a hard
        # segment boundary. Keep its in-window portion; the overlapping next
        # segment owns words that begin after this window.
        if start >= duration:
            continue
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
    return words


def _validate_transcript_coverage(transcript: str, words: list[ASRWord]) -> None:
    """Reject alignments that omit a material portion of the recognized text."""
    expected_count = len(re.findall(r"\S+", transcript))
    allowed_difference = max(2, round(expected_count * 0.2))
    if abs(expected_count - len(words)) > allowed_difference:
        raise QwenAlignmentError(
            "Qwen forced aligner word count materially diverges from the recognized transcript"
        )
