"""Orchestration of independently testable transcription stages."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from meeting_transcriber.alignment.chunks import ChunkMerger
from meeting_transcriber.alignment.speaker import (
    UNKNOWN_SPEAKER,
    OverlapSpeakerAligner,
    SpeakerAligner,
)
from meeting_transcriber.audio.chunker import AudioChunker
from meeting_transcriber.audio.preprocess import AudioPreprocessor
from meeting_transcriber.config import PipelineConfig
from meeting_transcriber.diarization.base import Diarizer
from meeting_transcriber.diarization.pyannote import PyannoteDiarizer
from meeting_transcriber.exporters.json import JsonTranscriptExporter
from meeting_transcriber.exporters.text import TextTranscriptExporter
from meeting_transcriber.models import (
    ASRRunMetadata,
    ASRWord,
    AttributedWord,
    AudioChunk,
    AudioMetadata,
    DiarizationSegment,
    PipelineResult,
    Transcript,
)
from meeting_transcriber.runtime.device import (
    peak_cuda_memory,
    reset_peak_cuda_memory,
    resolve_device,
)
from meeting_transcriber.runtime.lifecycle import release_model
from meeting_transcriber.transcription.base import BatchTranscriber, Transcriber
from meeting_transcriber.transcription.factory import create_transcriber
from meeting_transcriber.turns.builder import TurnBuilder

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


class ChunkProvider(Protocol):
    """Split a normalized WAV into ASR chunks."""

    def chunk(self, normalized_wav: Path) -> list[AudioChunk]:
        """Return chunked audio with absolute offsets."""


@dataclass(frozen=True)
class PreparedMeeting:
    """One normalization and diarization result reusable by multiple ASR runs."""

    metadata: AudioMetadata
    normalized_wav: Path
    chunks: list[AudioChunk]
    diarization: list[DiarizationSegment]
    work_directory: Path


class MeetingTranscriptionPipeline:
    """Run normalization, global diarization, chunked ASR, alignment, and export."""

    def __init__(
        self,
        config: PipelineConfig,
        diarizer_factory: Callable[[], Diarizer],
        transcriber_factory: Callable[[], Transcriber],
        preprocessor: AudioNormalizer | None = None,
        chunker: ChunkProvider | None = None,
        merger: ChunkMerger | None = None,
        aligner: SpeakerAligner | None = None,
        turn_builder: TurnBuilder | None = None,
    ) -> None:
        self.config = config
        self.diarizer_factory = diarizer_factory
        self.transcriber_factory = transcriber_factory
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.chunker = chunker or AudioChunker(
            config.resolved_chunk_duration, config.resolved_chunk_overlap
        )
        self.merger = merger or ChunkMerger()
        self.aligner = aligner or OverlapSpeakerAligner(config.alignment_tolerance)
        self.turn_builder = turn_builder or TurnBuilder(config.turn_gap_seconds)

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

    def prepare(self) -> PreparedMeeting:
        """Normalize and diarize once, returning reusable meeting state."""
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
        self._stage("chunk")
        chunks = self.chunker.chunk(normalized)
        LOGGER.info("created audio chunks", extra={"chunks": len(chunks)})
        return PreparedMeeting(metadata, normalized, chunks, diarization, job_directory)

    def cleanup(self, prepared: PreparedMeeting) -> None:
        """Remove temporary normalized audio unless debugging retained it."""
        if not self.config.keep_intermediate_files and prepared.work_directory.exists():
            shutil.rmtree(prepared.work_directory)

    def transcribe_prepared(
        self,
        prepared: PreparedMeeting,
        transcriber: Transcriber,
        backend: str,
        output_directory: Path,
    ) -> tuple[PipelineResult, ASRRunMetadata]:
        """Run one ASR backend over prepared meeting data and export its result."""
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
        transcriber.load()
        load_seconds = time.monotonic() - load_started
        words_by_chunk: dict[int, list[ASRWord]] = {}
        transcription_started = time.monotonic()
        try:
            if isinstance(transcriber, BatchTranscriber):
                words_by_chunk = transcriber.transcribe_chunks(prepared.chunks)
            else:
                for index, chunk in enumerate(prepared.chunks, start=1):
                    LOGGER.info(
                        "transcribing chunk",
                        extra={
                            "backend": backend,
                            "chunk": index,
                            "total_chunks": len(prepared.chunks),
                        },
                    )
                    words_by_chunk[chunk.chunk_id] = transcriber.transcribe(chunk)
            transcription_seconds = time.monotonic() - transcription_started
            allocated, reserved = peak_cuda_memory(
                getattr(transcriber, "device", self.config.device)
            )
        finally:
            release_model(transcriber)
        total_seconds = time.monotonic() - total_started
        self._stage("merge")
        words = self.merger.merge(prepared.chunks, words_by_chunk)
        self._stage("align")
        attributed = self.aligner.align(words, prepared.diarization)
        self._stage("turns")
        turns = self.turn_builder.build(attributed)
        speakers = sorted({word.speaker for word in attributed if word.speaker != UNKNOWN_SPEAKER})
        transcript = Transcript(
            metadata=prepared.metadata,
            asr_backend=backend,
            asr_model=transcriber.model_reference,
            diarization_model=self.config.pyannote_model,
            speakers=speakers,
            words=attributed,
            turns=turns,
        )
        result = PipelineResult(
            transcript=transcript,
            diarization=prepared.diarization,
            asr_words=words,
            output_directory=output_directory,
        )
        self._export(result, output_directory)
        metrics = ASRRunMetadata(
            backend=backend,
            model=transcriber.model_reference,
            device=getattr(transcriber, "device", self.config.device),
            dtype=transcriber.dtype_name,
            audio_duration_seconds=prepared.metadata.duration_seconds,
            chunk_duration_seconds=self.config.resolved_chunk_duration,
            chunk_overlap_seconds=self.config.resolved_chunk_overlap,
            model_load_seconds=load_seconds,
            transcription_seconds=transcription_seconds,
            total_asr_seconds=total_seconds,
            real_time_factor=total_seconds / prepared.metadata.duration_seconds,
            peak_cuda_memory_allocated_bytes=allocated,
            peak_cuda_memory_reserved_bytes=reserved,
            transformers_version=_package_version("transformers"),
            torch_version=_package_version("torch"),
            backend_metrics=getattr(transcriber, "backend_metrics", {}),
            backend_models=getattr(transcriber, "backend_models", {}),
        )
        LOGGER.info("ASR backend complete", extra=asdict(metrics))
        return result, metrics

    def _export(self, result: PipelineResult, output_directory: Path) -> None:
        self._stage("export")
        output_directory.mkdir(parents=True, exist_ok=True)
        JsonTranscriptExporter().export(result.transcript, output_directory / "transcript.json")
        TextTranscriptExporter().export(result.transcript, output_directory / "transcript.txt")
        if self.config.keep_intermediate_files:
            intermediate = output_directory / "intermediate"
            intermediate.mkdir(parents=True, exist_ok=True)
            self.write_records(intermediate / "diarization.json", result.diarization)
            self.write_records(intermediate / "asr_words.json", result.asr_words)
            self.write_records(intermediate / "attributed_words.json", result.transcript.words)

    @staticmethod
    def write_records(
        path: Path,
        records: Iterable[DiarizationSegment | ASRWord | AttributedWord],
    ) -> None:
        path.write_text(
            json.dumps([asdict(record) for record in records], indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _stage(name: str) -> None:
        LOGGER.info("pipeline stage", extra={"stage": name})


def create_default_pipeline(config: PipelineConfig) -> MeetingTranscriptionPipeline:
    """Construct the production adapter graph with no global model instances."""
    device = resolve_device(config.device)
    LOGGER.info("resolved runtime device", extra={"device": device, "backend": config.asr_backend})
    return MeetingTranscriptionPipeline(
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
