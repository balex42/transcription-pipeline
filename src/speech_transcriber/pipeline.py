"""Orchestration of independently testable transcription stages."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from speech_transcriber.audio.preprocess import AudioPreprocessor
from speech_transcriber.config import PipelineConfig
from speech_transcriber.diarization.base import Diarizer
from speech_transcriber.diarization.pyannote import PyannoteDiarizer
from speech_transcriber.finalization import TranscriptFinalizer, write_records
from speech_transcriber.models import (
    ASRRecognitionResult,
    ASRRunMetadata,
    AudioMetadata,
    NormalizedAudio,
    PipelineResult,
    RuntimeProvenance,
)
from speech_transcriber.prepared import PreparedRecording, sha256_file
from speech_transcriber.runtime.device import (
    peak_cuda_memory,
    reset_peak_cuda_memory,
    resolve_device,
)
from speech_transcriber.runtime.lifecycle import release_model
from speech_transcriber.transcription.base import Transcriber
from speech_transcriber.transcription.factory import create_transcriber

if TYPE_CHECKING:
    from speech_transcriber.alignment.speaker import SpeakerAligner
    from speech_transcriber.turns.builder import TurnBuilder

LOGGER = logging.getLogger(__name__)


def _package_version(package: str) -> str:
    """Return installed package provenance without contacting a registry."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


class AudioNormalizer(Protocol):
    """Normalize a source recording into the pipeline's PCM WAV contract."""

    def normalize(self, source: Path, destination: Path) -> AudioMetadata:
        """Return normalized recording metadata."""


class TranscriptionPipeline:
    """Run normalization, global diarization, ASR, alignment, and export."""

    def __init__(
        self,
        config: PipelineConfig,
        diarizer_factory: Callable[[], Diarizer],
        transcriber_factory: Callable[[], Transcriber],
        preprocessor: AudioNormalizer | None = None,
        aligner: SpeakerAligner | None = None,
        turn_builder: TurnBuilder | None = None,
        finalizer: TranscriptFinalizer | None = None,
    ) -> None:
        self.config = config
        self.diarizer_factory = diarizer_factory
        self.transcriber_factory = transcriber_factory
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.finalizer = finalizer or TranscriptFinalizer(config, aligner, turn_builder)

    def run(self) -> PipelineResult:
        """Execute the complete single-backend pipeline."""
        started = time.monotonic()
        prepared = self.prepare()
        try:
            result, _ = self.transcribe_prepared(
                prepared,
                self.transcriber_factory(),
                self.config.asr_backend,
                self.config.output_directory,
            )
            LOGGER.info(
                "pipeline complete", extra={"elapsed_seconds": round(time.monotonic() - started, 3)}
            )
            return result
        finally:
            self.cleanup(prepared)

    def prepare(self) -> PreparedRecording:
        """Normalize and diarize once, returning reusable recording state."""
        job_directory = self.config.working_directory / f"job-{int(time.time() * 1000)}"
        normalized = job_directory / "normalized.wav"
        self._stage("preprocess")
        metadata = self.preprocessor.normalize(self.config.input_path, normalized)
        LOGGER.info("audio normalized", extra={"duration_seconds": metadata.duration_seconds})
        self._stage("diarize")
        diarizer = self.diarizer_factory()
        try:
            diarization = diarizer.diarize(normalized)
        finally:
            release_model(diarizer)
        return PreparedRecording(
            NormalizedAudio(normalized, metadata),
            diarization,
            job_directory,
            normalized_audio_sha256=sha256_file(normalized),
            diarization_model=self.config.pyannote_model,
            language=self.config.language,
        )

    def cleanup(self, prepared: PreparedRecording) -> None:
        """Remove temporary normalized audio unless debugging retained it."""
        if (
            prepared.cleanup_enabled
            and not self.config.keep_intermediate_files
            and prepared.work_directory.exists()
        ):
            shutil.rmtree(prepared.work_directory)

    def transcribe_prepared(
        self,
        prepared: PreparedRecording,
        transcriber: Transcriber,
        backend: str,
        output_directory: Path,
    ) -> tuple[PipelineResult, ASRRunMetadata]:
        """Convenient local composition of recognition and backend-neutral finalization."""
        recognition = self.recognize_prepared(prepared, transcriber, backend)
        return self.finalize_prepared(prepared, recognition, output_directory), recognition.metadata

    def recognize_prepared(
        self,
        prepared: PreparedRecording,
        transcriber: Transcriber,
        backend: str,
    ) -> ASRRecognitionResult:
        """Run one backend and return portable ASR words with operational provenance."""
        self._stage("transcribe")
        reset_peak_cuda_memory(getattr(transcriber, "device", self.config.device))
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
            allocated, reserved = peak_cuda_memory(
                getattr(transcriber, "device", self.config.device)
            )
        finally:
            release_model(transcriber)
        total_seconds = time.monotonic() - total_started
        metadata = ASRRunMetadata(
            backend=backend,
            model=transcriber.model_reference,
            device=getattr(transcriber, "device", self.config.device),
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
            runtime=RuntimeProvenance(
                name="python",
                version=_package_version("speech-transcriber"),
                components={
                    "torch": _package_version("torch"),
                    "transformers": _package_version("transformers"),
                },
            ),
            backend_metrics=getattr(transcriber, "backend_metrics", {}),
            backend_models=getattr(transcriber, "backend_models", {}),
            backend_configuration=getattr(transcriber, "backend_configuration", {}),
        )
        LOGGER.info("ASR backend complete", extra=asdict(metadata))
        return ASRRecognitionResult(words=words, metadata=metadata)

    def finalize_prepared(
        self,
        prepared: PreparedRecording,
        recognition: ASRRecognitionResult,
        output_directory: Path,
    ) -> PipelineResult:
        """Delegate transcript construction to the backend-neutral finalizer."""
        return self.finalizer.finalize_prepared(prepared, recognition, output_directory)

    @staticmethod
    def write_records(
        path: Path,
        records: object,
    ) -> None:
        """Retain the legacy helper while implementation lives in finalization."""
        write_records(path, records)  # type: ignore[arg-type]

    @staticmethod
    def _stage(name: str) -> None:
        LOGGER.info("pipeline stage", extra={"stage": name})


def create_default_pipeline(config: PipelineConfig) -> TranscriptionPipeline:
    """Construct the production adapter graph with no global model instances."""
    device = resolve_device(config.device)
    LOGGER.info("resolved runtime device", extra={"device": device, "backend": config.asr_backend})
    return TranscriptionPipeline(
        config=config,
        diarizer_factory=lambda: PyannoteDiarizer(
            config.pyannote_model,
            device,
            config.num_speakers,
            config.min_speakers,
            config.max_speakers,
        ),
        transcriber_factory=lambda: create_transcriber(config, device),
    )
