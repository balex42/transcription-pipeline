"""Backend-neutral recognition orchestration for heterogeneous ASR runtimes.

``RecognitionRunner`` owns the generic concerns of running one ``Transcriber``
against a prepared recording and producing the canonical ASR artifact. It must
never import PyTorch, Transformers, pyannote, audio normalization, or any
backend implementation, so the dedicated faster-whisper image can run
recognition without the Transformers ASR stack installed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol

from speech_transcriber.models import (
    ASRRecognitionResult,
    ASRRunMetadata,
    RuntimeProvenance,
)
from speech_transcriber.prepared import PreparedRecording
from speech_transcriber.transcription.base import Transcriber

LOGGER = logging.getLogger(__name__)


class MemoryMetrics(Protocol):
    """Optional backend-provided CUDA memory accounting."""

    def reset(self) -> None:
        """Reset per-run peak accounting before model load."""

    def peak(self) -> tuple[int | None, int | None]:
        """Return allocated and reserved peak bytes for the current run."""


def _package_version(package: str) -> str:
    """Return installed package provenance without contacting a registry."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


class RecognitionRunner:
    """Run one backend and return portable ASR words with operational provenance."""

    def __init__(self, memory_metrics: MemoryMetrics | None = None) -> None:
        self.memory_metrics = memory_metrics

    def recognize(
        self,
        prepared: PreparedRecording,
        transcriber: Transcriber,
        backend: str,
    ) -> ASRRecognitionResult:
        """Load, transcribe, and release one backend against a prepared recording."""
        device = getattr(transcriber, "device", "cpu")
        if self.memory_metrics is not None:
            self.memory_metrics.reset()
        total_started = time.monotonic()
        load_started = time.monotonic()
        LOGGER.info(
            "loading ASR backend",
            extra={
                "backend": backend,
                "model": transcriber.model_reference,
                "dtype": transcriber.dtype_name,
            },
        )
        try:
            transcriber.load()
            load_seconds = time.monotonic() - load_started
            transcription_started = time.monotonic()
            words = transcriber.transcribe(prepared.audio)
            transcription_seconds = time.monotonic() - transcription_started
            allocated, reserved = (
                self.memory_metrics.peak() if self.memory_metrics is not None else (None, None)
            )
        finally:
            transcriber.release()
        total_seconds = time.monotonic() - total_started
        runtime = getattr(transcriber, "runtime_provenance", None)
        if not isinstance(runtime, RuntimeProvenance):
            runtime = RuntimeProvenance(
                name="python",
                version=_package_version("speech-transcriber"),
                components={
                    "torch": _package_version("torch"),
                    "transformers": _package_version("transformers"),
                },
            )
        metadata = ASRRunMetadata(
            backend=backend,
            model=transcriber.model_reference,
            device=device,
            dtype=transcriber.dtype_name,
            audio_duration_seconds=prepared.audio.metadata.duration_seconds,
            model_load_seconds=load_seconds,
            transcription_seconds=transcription_seconds,
            total_asr_seconds=total_seconds,
            real_time_factor=total_seconds / prepared.audio.metadata.duration_seconds,
            peak_cuda_memory_allocated_bytes=allocated,
            peak_cuda_memory_reserved_bytes=reserved,
            normalized_audio_sha256=prepared.normalized_audio_sha256,
            transformers_version=_package_version("transformers"),
            torch_version=_package_version("torch"),
            runtime=runtime,
            backend_metrics=getattr(transcriber, "backend_metrics", {}),
            backend_models=getattr(transcriber, "backend_models", {}),
            backend_configuration=getattr(transcriber, "backend_configuration", {}),
        )
        LOGGER.info("ASR backend complete", extra=asdict(metadata))
        return ASRRecognitionResult(words=words, metadata=metadata)
