"""Typed data contracts shared by pipeline stages.

All public timestamps are floating-point seconds from the start of the source
recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AudioMetadata:
    """Metadata for the normalized recording."""

    source: str
    duration_seconds: float
    sample_rate: int = 16_000
    channels: int = 1
    sample_width_bits: int = 16


@dataclass(frozen=True)
class NormalizedAudio:
    """One normalized recording passed intact to an ASR backend."""

    path: Path
    metadata: AudioMetadata


@dataclass(frozen=True)
class AudioSegment:
    """Backend-private PCM segment with absolute recording offsets."""

    index: int
    start: float
    end: float
    audio: NDArray[np.float32]
    sample_rate: int = 16_000


@dataclass(frozen=True)
class DiarizationSegment:
    """One exclusive diarization region for an anonymous speaker."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class ASRWord:
    """Backend-neutral lexical output with optional native word boundaries."""

    text: str
    end: float
    start: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class AttributedWord:
    """An ASR word assigned to a diarization speaker."""

    text: str
    start: float
    end: float
    speaker: str
    start_is_inferred: bool = False


@dataclass(frozen=True)
class SpeakerTurn:
    """A contiguous derived run of attributed words."""

    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """Canonical transcript data before serialization."""

    metadata: AudioMetadata
    asr_backend: str
    asr_model: str
    diarization_model: str
    speakers: list[str]
    words: list[AttributedWord]
    turns: list[SpeakerTurn]
    language: str = "de-DE"
    version: str = "1.0"


@dataclass(frozen=True)
class PipelineResult:
    """Outputs retained from one complete pipeline execution."""

    transcript: Transcript
    diarization: list[DiarizationSegment] = field(default_factory=list)
    asr_words: list[ASRWord] = field(default_factory=list)
    output_directory: Path | None = None


@dataclass(frozen=True)
class ASRRunMetadata:
    """Operational metadata from one backend execution, not a quality score."""

    backend: str
    model: str
    device: str
    dtype: str
    audio_duration_seconds: float
    model_load_seconds: float
    transcription_seconds: float
    total_asr_seconds: float
    real_time_factor: float
    peak_cuda_memory_allocated_bytes: int | None
    peak_cuda_memory_reserved_bytes: int | None
    transformers_version: str = "unknown"
    torch_version: str = "unknown"
    runtime: RuntimeProvenance = field(default_factory=lambda: RuntimeProvenance())
    backend_metrics: dict[str, float] = field(default_factory=dict)
    backend_models: dict[str, str] = field(default_factory=dict)
    backend_configuration: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeProvenance:
    """Runtime details for a recognition implementation, independent of a Python API."""

    name: str = "python"
    version: str = "unknown"
    components: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ASRRecognitionResult:
    """Backend-neutral ASR output ready for independent transcript finalization."""

    words: list[ASRWord]
    metadata: ASRRunMetadata
