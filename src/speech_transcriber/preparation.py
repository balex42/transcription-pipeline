"""Preparation stage: normalization, diarization, and the prepared artifact.

Preparation is the first worker stage of the split pipeline. It owns only the
Transformers-runtime work that other stages must never repeat: audio
normalization, one pyannote diarization pass, diarization model release, and
construction of the versioned ``PreparedRecording`` artifact. Recognition and
finalization belong to the separate backend-neutral runners.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from speech_transcriber.audio.preprocess import AudioPreprocessor
from speech_transcriber.config import PipelineConfig
from speech_transcriber.diarization.base import Diarizer
from speech_transcriber.diarization.pyannote import PyannoteDiarizer
from speech_transcriber.models import AudioMetadata, NormalizedAudio
from speech_transcriber.prepared import PreparedRecording, sha256_file
from speech_transcriber.runtime.device import resolve_device
from speech_transcriber.runtime.lifecycle import release_model

LOGGER = logging.getLogger(__name__)


class AudioNormalizer(Protocol):
    """Normalize a source recording into the pipeline's PCM WAV contract."""

    def normalize(self, source: Path, destination: Path) -> AudioMetadata:
        """Return normalized recording metadata."""


class PreparationRunner:
    """Normalize and diarize once, producing the reusable prepared artifact."""

    def __init__(
        self,
        config: PipelineConfig,
        diarizer_factory: Callable[[], Diarizer],
        preprocessor: AudioNormalizer | None = None,
    ) -> None:
        self.config = config
        self.diarizer_factory = diarizer_factory
        self.preprocessor = preprocessor or AudioPreprocessor()

    def prepare(self) -> PreparedRecording:
        """Normalize the source recording and diarize the normalized WAV."""
        job_directory = self.config.working_directory / f"job-{int(time.time() * 1000)}"
        normalized = job_directory / "normalized.wav"
        LOGGER.info("pipeline stage", extra={"stage": "preprocess"})
        metadata = self.preprocessor.normalize(self.config.input_path, normalized)
        LOGGER.info("audio normalized", extra={"duration_seconds": metadata.duration_seconds})
        LOGGER.info("pipeline stage", extra={"stage": "diarize"})
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

    @classmethod
    def create_default(cls, config: PipelineConfig) -> PreparationRunner:
        """Construct the production pyannote runner with the resolved device."""
        device = resolve_device(config.device)
        LOGGER.info("resolved runtime device", extra={"device": device})
        return cls(
            config,
            diarizer_factory=lambda: PyannoteDiarizer(
                config.pyannote_model,
                device,
                config.num_speakers,
                config.min_speakers,
                config.max_speakers,
            ),
        )