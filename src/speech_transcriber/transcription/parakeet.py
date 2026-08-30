"""NVIDIA NeMo Parakeet TDT adapter with native word timestamps.

The adapter restores the trusted ``.nemo`` checkpoint from an explicit local
path or a resolved Hugging Face cache snapshot. It deliberately never calls
``from_pretrained()`` during recognition, so air-gapped runs cannot fall back
to an online model lookup, and it does not use the previous Transformers
adapter path.

Recognition keeps the established long-form strategy: the normalized recording
is split into overlapping 180-second internal segments with the shared
segmenter, each segment's normalized float32 samples are transcribed in one
NeMo call with native word timestamps, and overlap words are reconciled into
recording-global positions by the shared segment reconciliation. No forced
alignment and no text-derived timestamps are used.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from speech_transcriber.audio.segmenter import AudioSegmenter
from speech_transcriber.config import PARAKEET_MODEL_FILE
from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.models import ASRWord, AudioSegment, NormalizedAudio
from speech_transcriber.transcription import nemo_support
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities
from speech_transcriber.transcription.segments import reconcile_segment_words

LOGGER = logging.getLogger(__name__)

TIMESTAMP_TOLERANCE_SECONDS = 1e-3


class ParakeetTranscriber(Transcriber):
    """Generate punctuation-preserving words and native TDT timestamps."""

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
        # The checkpoint controls its precision; do not infer one from the
        # process-wide PyTorch default or force a conversion.
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
            "checkpoint_file": PARAKEET_MODEL_FILE,
        }
        self.runtime_provenance = nemo_support.initial_runtime_provenance()

    def load(self) -> None:
        """Load the NeMo Parakeet model lazily."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Segment and reconcile the full recording inside the Parakeet adapter."""
        segments = self._segmenter.segment(audio)
        self.backend_metrics["segments_processed"] = float(len(segments))
        return reconcile_segment_words(
            segments, {segment.index: self._transcribe_segment(segment) for segment in segments}
        )

    def _transcribe_segment(self, segment: AudioSegment) -> list[ASRWord]:
        """Return native start/end timestamps relative to one internal segment."""
        model = self._load()
        words = flatten_parakeet_words(
            self._transcribe(model, segment),
            tolerance=TIMESTAMP_TOLERANCE_SECONDS,
        )
        return validate_parakeet_segment_words(words, segment)

    def _transcribe(self, model: object, segment: AudioSegment) -> Sequence[object]:
        try:
            outputs: Sequence[object] = model.transcribe(  # type: ignore[attr-defined]
                [segment.audio],
                batch_size=1,
                return_hypotheses=True,
                timestamps=True,
            )
            return outputs
        except Exception as error:
            raise ASROutputError(
                f"Parakeet recognition failed for segment {segment.index}: {error}"
            ) from error

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        checkpoint, snapshot = nemo_support.resolve_checkpoint_path(
            self.model_reference, PARAKEET_MODEL_FILE, subject="Parakeet model"
        )
        try:
            self._model = _restore_parakeet_model(checkpoint, self.device)
        except Exception as error:
            raise ModelLoadError(
                f"could not load Parakeet model {self.model_reference}: {error}"
            ) from error
        self.backend_models["model_file"] = PARAKEET_MODEL_FILE
        if snapshot:
            self.backend_models["model_snapshot"] = snapshot
        LOGGER.info("loading Parakeet from snapshot", extra={"snapshot": snapshot or "direct"})
        self.runtime_provenance = nemo_support.nemo_runtime_provenance()
        return self._model

    def release(self) -> None:
        """Drop model references for sequential GPU execution."""
        self._model = None


def resolve_parakeet_model_path(model: str) -> str:
    """Locate Parakeet's ``.nemo`` checkpoint without a runtime Hub lookup.

    A configured local file or directory takes precedence. Repository IDs are
    resolved from the active Hugging Face cache through the shared snapshot
    helper via ``refs/main``. Resolution is always strict: runtime model
    downloading is intentionally unsupported, so a missing or ambiguous cache
    fails instead of returning a Hub repository ID.
    """
    checkpoint, _snapshot = nemo_support.resolve_checkpoint_path(
        model, PARAKEET_MODEL_FILE, subject="Parakeet model"
    )
    return checkpoint


def _restore_parakeet_model(model_path: str, device: str) -> object:
    """Restore and place Parakeet on the requested device, importing NeMo lazily."""
    return nemo_support.restore_model(model_path, device, subject="Parakeet")


def flatten_parakeet_words(
    outputs: Sequence[object],
    *,
    tolerance: float = TIMESTAMP_TOLERANCE_SECONDS,
) -> list[ASRWord]:
    """Map ``Hypothesis.timestamp['word']`` records to canonical ASR words.

    The checkpoint transcribes one segment per call, so NeMo's local word
    timestamps are segment-local and are rebased by the caller. Output is
    validated for numeric bounds and ordering; punctuation and capitalization
    are preserved exactly as emitted.
    """
    if len(outputs) != 1:
        raise ASROutputError(f"Parakeet returned {len(outputs)} hypotheses for one segment")
    try:
        records = nemo_support.word_timestamp_records(outputs[0], subject="Parakeet")
    except ValueError as error:
        raise ASROutputError(str(error)) from error

    words: list[ASRWord] = []
    previous_end: float | None = None
    for record in records:
        if not isinstance(record, Mapping):
            raise ASROutputError("Parakeet word timestamp must be an object")
        text = record.get("word")
        start = record.get("start")
        end = record.get("end")
        if not isinstance(text, str) or not text.strip():
            raise ASROutputError("Parakeet word timestamp is missing text")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ASROutputError("Parakeet word timestamp is missing numeric start/end values")
        if start < -tolerance or end < start - tolerance:
            raise ASROutputError(
                f"Parakeet word timestamp is invalid: '{text}' {start}-{end}"
            )
        if previous_end is not None and start < previous_end - tolerance:
            raise ASROutputError(
                "Parakeet word timestamps reorder at index "
                f"{len(words)}: '{text}' starts {start} after {previous_end}"
            )
        if previous_end is None or end > previous_end:
            previous_end = float(end)
        confidence_value = record.get("confidence")
        # NeMo's Parakeet TDT word records expose no stable per-word confidence.
        confidence = (
            float(confidence_value) if isinstance(confidence_value, int | float) else None
        )
        words.append(
            ASRWord(text=text.strip(), start=float(start), end=float(end), confidence=confidence)
        )
    return words


def validate_parakeet_segment_words(
    words: Sequence[ASRWord], segment: AudioSegment
) -> list[ASRWord]:
    """Validate segment-local timestamps before the shared global reconciliation."""
    duration = segment.end - segment.start
    for word in words:
        if word.end > duration + 1.0:
            raise ASROutputError(
                "Parakeet word timestamp exceeds the segment duration: "
                f"'{word.text}' ends at {word.end}"
            )
    return list(words)