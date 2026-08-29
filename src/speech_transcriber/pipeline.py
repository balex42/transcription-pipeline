"""Orchestration of independently testable transcription stages."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
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
)
from speech_transcriber.prepared import PreparedRecording, sha256_file
from speech_transcriber.recognition import RecognitionRunner
from speech_transcriber.runtime.device import (
    TorchMemoryMetrics,
    resolve_device,
)
from speech_transcriber.runtime.lifecycle import release_model
from speech_transcriber.transcription.base import Transcriber
from speech_transcriber.transcription.factory import create_transcriber

if TYPE_CHECKING:
    from speech_transcriber.alignment.speaker import SpeakerAligner
    from speech_transcriber.turns.builder import TurnBuilder

LOGGER = logging.getLogger(__name__)


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
        recognition_runner: RecognitionRunner | None = None,
    ) -> None:
        self.config = config
        self.diarizer_factory = diarizer_factory
        self.transcriber_factory = transcriber_factory
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.finalizer = finalizer or TranscriptFinalizer(config, aligner, turn_builder)
        self.recognition_runner = recognition_runner or RecognitionRunner()

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
        """Delegate recognition to the backend-neutral runner."""
        self._stage("transcribe")
        return self.recognition_runner.recognize(prepared, transcriber, backend)

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
        recognition_runner=RecognitionRunner(TorchMemoryMetrics(device)),
    )
