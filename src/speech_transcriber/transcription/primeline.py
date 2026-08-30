"""NVIDIA NeMo Parakeet TDT adapter for the Primeline German speech checkpoint.

Primeline is distributed as a NeMo ``.nemo`` checkpoint and runs on the shared
NeMo ASR runtime alongside Parakeet and Canary while remaining a fully
independent adapter. The trusted checkpoint is restored from an explicit local
path or a resolved Hugging Face cache snapshot; runtime recognition never calls
``from_pretrained()`` or contacts the Hub.

Recognition uses one invariant path: the whole normalized WAV is transcribed in
a single NeMo call with native word timestamps, relying on the checkpoint's
local-attention long-form behavior. No segmentation, chunking, forced
alignment, or text-derived timestamps are used.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from speech_transcriber.config import PRIMELINE_MODEL_FILE
from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.model_cache import resolve_hf_snapshot, snapshot_revision
from speech_transcriber.models import ASRWord, NormalizedAudio
from speech_transcriber.transcription import nemo_support
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities

LOGGER = logging.getLogger(__name__)

TIMESTAMP_TOLERANCE_SECONDS = 1e-3


class PrimelineTranscriber(Transcriber):
    """Transcribe one normalized recording with NeMo Primeline and native timestamps."""

    capabilities = TranscriberCapabilities(True, True, True, True)

    def __init__(self, model: str, device: str) -> None:
        self.model_reference = model
        self.device = device
        # The checkpoint controls its precision; do not infer one from the
        # process-wide PyTorch default or force a conversion.
        self.dtype_name = "checkpoint-default"
        self._model: object | None = None
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        self.backend_configuration: dict[str, str | int | float | bool | None] = {
            "timestamps": True,
            "batch_size": 1,
            "inference_mode": "single_pass_local_attention",
            "checkpoint_file": PRIMELINE_MODEL_FILE,
        }
        self.runtime_provenance = nemo_support.initial_runtime_provenance()

    def load(self) -> None:
        """Restore the local NeMo checkpoint without contacting Hugging Face."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Run one whole-recording NeMo transcribe call with native timestamps."""
        model = self._load()
        try:
            outputs = self._transcribe(model, audio_path=audio.path)
            words = flatten_primeline_words(
                outputs,
                duration_seconds=audio.metadata.duration_seconds,
            )
        except ASROutputError:
            raise
        except Exception as error:
            raise ASROutputError(
                "Primeline recognition failed for "
                f"{audio.path} with model {self.model_reference}: {error}"
            ) from error
        self.backend_metrics["word_count"] = float(len(words))
        return words

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        model_path = resolve_primeline_model_path(self.model_reference)
        try:
            self._model = _restore_primeline_model(model_path, self.device)
        except Exception as error:
            raise ModelLoadError(
                f"could not load Primeline model {self.model_reference}: {error}"
            ) from error
        self.backend_models["model_file"] = Path(model_path).name
        if Path(model_path).parent.parent.name == "snapshots":
            self.backend_models["model_snapshot"] = (
                snapshot_revision(Path(model_path).parent) or "unknown"
            )
        LOGGER.info(
            "loading Primeline from cached snapshot revision %.12s",
            snapshot_revision(Path(model_path).parent) or "unknown",
        )
        self.runtime_provenance = nemo_support.nemo_runtime_provenance()
        return self._model

    def _transcribe(self, model: Any, *, audio_path: Path) -> Any:
        try:
            return model.transcribe(
                [str(audio_path)],
                batch_size=1,
                return_hypotheses=True,
                timestamps=True,
            )
        except Exception as error:
            raise ASROutputError(
                f"Primeline recognition failed for {audio_path}: {error}"
            ) from error

    def release(self) -> None:
        """Drop the NeMo model so the runner can release GPU memory between jobs."""
        self._model = None


def resolve_primeline_model_path(model: str) -> str:
    """Locate Primeline's ``.nemo`` checkpoint without a runtime Hub lookup.

    A configured local file or directory takes precedence. Repository IDs are
    resolved from the active Hugging Face cache through the shared snapshot
    helper via ``refs/main``. Resolution is always strict: runtime model
    downloading is intentionally unsupported, so a missing or ambiguous cache
    fails instead of returning a Hub repository ID.
    """
    configured = Path(model).expanduser()
    if configured.is_absolute() or configured.exists():
        return _primeline_model_file(configured)

    snapshot = resolve_hf_snapshot(model, subject="Primeline model")
    return _primeline_model_file(snapshot)


def flatten_primeline_words(
    outputs: Sequence[Any],
    *,
    duration_seconds: float | None = None,
    tolerance: float = TIMESTAMP_TOLERANCE_SECONDS,
) -> list[ASRWord]:
    """Map ``Hypothesis.timestamp['word']`` records to canonical ASR words.

    The checkpoint transcribes the whole recording in one call, so NeMo's
    local word timestamps are already recording-global; no rebasing is
    applied. Output is validated for numeric bounds, ordering, and duration.
    """
    if len(outputs) != 1:
        raise ASROutputError(f"Primeline returned {len(outputs)} hypotheses for one recording")
    timestamp = getattr(outputs[0], "timestamp", None)
    if not isinstance(timestamp, Mapping):
        raise ASROutputError("Primeline output is missing timestamp metadata")
    records = timestamp.get("word")
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ASROutputError("Primeline output is missing word timestamps")

    words: list[ASRWord] = []
    previous_end: float | None = None
    for record in records:
        if not isinstance(record, Mapping):
            raise ASROutputError("Primeline word timestamp must be an object")
        text = record.get("word")
        start = record.get("start")
        end = record.get("end")
        if not isinstance(text, str) or not text.strip():
            raise ASROutputError("Primeline word timestamp is missing text")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ASROutputError("Primeline word timestamp is missing numeric start/end values")
        if start < -tolerance or end < start - tolerance:
            raise ASROutputError(
                f"Primeline word timestamp is invalid: '{text}' {start}-{end}"
            )
        if previous_end is not None and start < previous_end - tolerance:
            raise ASROutputError(
                "Primeline word timestamps reorder at index "
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

    if duration_seconds is not None:
        for word in words:
            if word.end > duration_seconds + 1.0:
                raise ASROutputError(
                    f"Primeline word timestamp exceeds recording duration: "
                    f"'{word.text}' ends at {word.end}"
                )
    return words


def _primeline_model_file(path: Path) -> str:
    return nemo_support.model_file(path, PRIMELINE_MODEL_FILE, subject="Primeline model")


def _restore_primeline_model(model_path: str, device: str) -> Any:
    """Restore and place Primeline on the requested device, importing NeMo lazily."""
    return nemo_support.restore_model(model_path, device, subject="Primeline")