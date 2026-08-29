"""Sequential manual ASR comparison built on one prepared recording."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from speech_transcriber.asr_artifact import write_asr_result_files
from speech_transcriber.config import ASR_BACKENDS, PipelineConfig
from speech_transcriber.errors import UnsupportedASRBackendError
from speech_transcriber.pipeline import TranscriptionPipeline
from speech_transcriber.runtime.device import resolve_device
from speech_transcriber.transcription.base import Transcriber
from speech_transcriber.transcription.factory import create_transcriber

LOGGER = logging.getLogger(__name__)


class ASRComparisonRunner:
    """Prepare audio/diarization once and run requested ASR backends sequentially."""

    def __init__(
        self,
        pipeline: TranscriptionPipeline,
        transcriber_factory: Callable[[PipelineConfig, str], Transcriber] = create_transcriber,
    ) -> None:
        self.pipeline = pipeline
        self.transcriber_factory = transcriber_factory

    def run(self, backends: list[str], output_directory: Path) -> None:
        """Write side-by-side backend transcripts and operational metadata."""
        invalid = sorted(set(backends) - set(ASR_BACKENDS))
        if invalid:
            raise UnsupportedASRBackendError(f"unsupported ASR backend(s): {', '.join(invalid)}")
        if not backends:
            raise ValueError("at least one ASR backend is required")
        output_directory.mkdir(parents=True, exist_ok=True)
        prepared = self.pipeline.prepare()
        try:
            self.pipeline.write_records(output_directory / "diarization.json", prepared.diarization)
            device = resolve_device(self.pipeline.config.device)
            runs = []
            for index, backend in enumerate(backends, start=1):
                backend_config = replace(self.pipeline.config, asr_backend=backend, asr_model=None)
                LOGGER.info(
                    "running comparison backend",
                    extra={"index": index, "total": len(backends), "backend": backend},
                )
                recognition = self.pipeline.recognize_prepared(
                    prepared,
                    self.transcriber_factory(backend_config, device),
                    backend,
                )
                self.pipeline.finalize_prepared(
                    prepared,
                    recognition,
                    output_directory / backend,
                )
                write_asr_result_files(recognition, output_directory / backend)
                runs.append(asdict(recognition.metadata))
            metadata = {
                "source": prepared.audio.metadata.source,
                "audio_duration_seconds": prepared.audio.metadata.duration_seconds,
                "diarization_model": self.pipeline.config.pyannote_model,
                "asr_runs": runs,
            }
            (output_directory / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            self.pipeline.cleanup(prepared)
