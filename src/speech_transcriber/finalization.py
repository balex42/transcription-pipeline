"""Backend-neutral transcript alignment, turn building, and export."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from speech_transcriber.alignment.speaker import (
    UNKNOWN_SPEAKER,
    OverlapSpeakerAligner,
    SpeakerAligner,
)
from speech_transcriber.config import FinalizationConfig
from speech_transcriber.exporters.json import JsonTranscriptExporter
from speech_transcriber.exporters.text import TextTranscriptExporter
from speech_transcriber.models import (
    ASRRecognitionResult,
    ASRWord,
    AttributedWord,
    DiarizationSegment,
    PipelineResult,
    Transcript,
)
from speech_transcriber.prepared import PreparedRecording
from speech_transcriber.turns.builder import TurnBuilder

LOGGER = logging.getLogger(__name__)


class TranscriptFinalizer:
    """Finalize portable prepared and ASR artifacts without model runtime dependencies."""

    def __init__(
        self,
        config: FinalizationConfig,
        aligner: SpeakerAligner | None = None,
        turn_builder: TurnBuilder | None = None,
    ) -> None:
        self.config = config
        self.aligner = aligner or OverlapSpeakerAligner(config.alignment_tolerance)
        self.turn_builder = turn_builder or TurnBuilder(config.turn_gap_seconds)

    def finalize_prepared(
        self,
        prepared: PreparedRecording,
        recognition: ASRRecognitionResult,
        output_directory: Path,
        *,
        expected_backend: str | None = None,
    ) -> PipelineResult:
        """Build and export a transcript from prepared and backend-neutral ASR artifacts."""
        if expected_backend is not None and recognition.metadata.backend != expected_backend:
            raise ValueError(
                f"ASR artifact backend {recognition.metadata.backend!r} does not match "
                f"{expected_backend!r}"
            )
        if not math.isclose(
            recognition.metadata.audio_duration_seconds,
            prepared.audio.metadata.duration_seconds,
            rel_tol=0.0,
            abs_tol=0.001,
        ):
            raise ValueError("ASR artifact audio duration does not match the prepared recording")
        if recognition.metadata.normalized_audio_sha256 != prepared.normalized_audio_sha256:
            raise ValueError(
                "ASR artifact normalized audio SHA-256 does not match the prepared recording"
            )

        self._stage("align")
        attributed = self.aligner.align(recognition.words, prepared.diarization)
        self._stage("turns")
        turns = self.turn_builder.build(attributed)
        speakers = sorted({word.speaker for word in attributed if word.speaker != UNKNOWN_SPEAKER})
        # The prepared artifact owns its provenance; finalization never fabricates it.
        transcript = Transcript(
            metadata=prepared.audio.metadata,
            asr_backend=recognition.metadata.backend,
            asr_model=recognition.metadata.model,
            diarization_model=prepared.diarization_model,
            speakers=speakers,
            words=attributed,
            turns=turns,
            language=prepared.language,
        )
        result = PipelineResult(
            transcript=transcript,
            diarization=prepared.diarization,
            asr_words=recognition.words,
            output_directory=output_directory,
        )
        self._export(result, output_directory)
        return result

    def _export(self, result: PipelineResult, output_directory: Path) -> None:
        self._stage("export")
        output_directory.mkdir(parents=True, exist_ok=True)
        JsonTranscriptExporter().export(result.transcript, output_directory / "transcript.json")
        TextTranscriptExporter().export(result.transcript, output_directory / "transcript.txt")
        if self.config.keep_intermediate_files:
            intermediate = output_directory / "intermediate"
            intermediate.mkdir(parents=True, exist_ok=True)
            write_records(intermediate / "diarization.json", result.diarization)
            write_records(intermediate / "asr_words.json", result.asr_words)
            write_records(intermediate / "attributed_words.json", result.transcript.words)

    @staticmethod
    def _stage(name: str) -> None:
        LOGGER.info("pipeline stage", extra={"stage": name})


def write_records(
    path: Path,
    records: Iterable[DiarizationSegment | ASRWord | AttributedWord],
) -> None:
    """Write portable intermediate records for local debugging."""
    path.write_text(
        json.dumps(
            [asdict(record) for record in records],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
