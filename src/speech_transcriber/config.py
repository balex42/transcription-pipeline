"""Stage-specific configuration with CLI, environment, then default precedence.

Each worker stage parses only its own settings: invalid environment values for
other stages cannot break a command that never reads them. Shared defaults and
validators live here; the stage dataclasses are the only configuration types.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
DEFAULT_NEMOTRON_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS = 13
DEFAULT_VOXTRAL_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
DEFAULT_VOXTRAL_DELAY_MS = 2400
DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS = 4
VOXTRAL_MAX_DELAY_MS = 2400
VOXTRAL_MIN_DELAY_MS = 80
VOXTRAL_DELAY_STEP_MS = 80
VOXTRAL_MAX_TIMESTAMP_OFFSET_TOKENS = 30
DEFAULT_FASTER_WHISPER_MODEL = "Systran/faster-whisper-large-v3"
DEFAULT_FASTER_WHISPER_COMPUTE_TYPE = "float16"
FASTER_WHISPER_COMPUTE_TYPES = ("float16", "bfloat16", "float32", "int8", "int8_float16")
DEFAULT_PRIMELINE_MODEL = "primeline/parakeet-primeline"
DEFAULT_CANARY_MODEL = "nvidia/canary-1b-v2"
DEFAULT_CANARY_CHUNK_DURATION_SECONDS = 10.0
PRIMELINE_MODEL_FILE = "2_95_WER.nemo"
PARAKEET_MODEL_FILE = "parakeet-tdt-0.6b-v3.nemo"
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
ASR_BACKENDS = (
    "parakeet",
    "primeline",
    "qwen",
    "nemotron",
    "voxtral",
    "faster-whisper",
    "canary",
)
ASRRuntime = Literal["transformers", "nemo", "ctranslate2"]
BACKEND_RUNTIMES: Final[dict[str, ASRRuntime]] = {
    "parakeet": "nemo",
    "primeline": "nemo",
    "canary": "nemo",
    "qwen": "transformers",
    "nemotron": "transformers",
    "voxtral": "transformers",
    "faster-whisper": "ctranslate2",
}
QWEN_MAX_ALIGNMENT_DURATION_SECONDS = 300.0
DEFAULT_PARAKEET_SEGMENT_DURATION = 180.0
DEFAULT_PARAKEET_SEGMENT_OVERLAP = 15.0
DEFAULT_QWEN_SEGMENT_DURATION = 240.0
DEFAULT_QWEN_SEGMENT_OVERLAP = 15.0
DEFAULT_ASR_MODELS = {
    "parakeet": DEFAULT_PARAKEET_MODEL,
    "primeline": DEFAULT_PRIMELINE_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
    "nemotron": DEFAULT_NEMOTRON_MODEL,
    "voxtral": DEFAULT_VOXTRAL_MODEL,
    "faster-whisper": DEFAULT_FASTER_WHISPER_MODEL,
    "canary": DEFAULT_CANARY_MODEL,
}
DEFAULT_ALIGNMENT_TOLERANCE = 0.25
DEFAULT_TURN_GAP_SECONDS = 1.0
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_WORKING_DIRECTORY = "work"
DEFAULT_LANGUAGE = "de-DE"


class _Choices:
    """Resolve stage settings from CLI overrides, environment, then defaults."""

    def __init__(self, overrides: Mapping[str, object], env: Mapping[str, str]) -> None:
        self.overrides = overrides
        self.env = env

    def string(self, name: str, env_name: str, default: str) -> str:
        value = self.overrides.get(name)
        return str(value) if value is not None else self.env.get(env_name, default)

    def string_or_none(self, name: str, env_name: str | None) -> str | None:
        value = self.overrides.get(name)
        if value is not None:
            return str(value)
        if env_name is None:
            return None
        return self.env.get(env_name) or None

    def float_or_default(self, name: str, env_name: str | None, default: float) -> float:
        value = self.overrides.get(name)
        if value is None:
            text = self.env.get(env_name) if env_name else None
            return float(text) if text else default
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise ValueError(f"{name} must be numeric")
        return float(value)

    def int_or_none(self, name: str, env_name: str | None) -> int | None:
        value = self.overrides.get(name)
        if value is None:
            text = self.env.get(env_name) if env_name else None
            return int(text) if text else None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    def int_or_default(self, name: str, env_name: str, default: int) -> int:
        value = self.int_or_none(name, env_name)
        return default if value is None else value


def _working_directory(overrides: Mapping[str, object], env: Mapping[str, str]) -> Path:
    value = overrides.get("working_directory")
    text = str(value) if value is not None else env.get("WORKING_DIRECTORY")
    return Path(text) if text else Path.cwd() / DEFAULT_WORKING_DIRECTORY


def validate_qwen_segment_duration(duration: float) -> None:
    """Reject Qwen segments that exceed the forced aligner's audio limit."""
    if duration > QWEN_MAX_ALIGNMENT_DURATION_SECONDS:
        raise ValueError(
            "Qwen segment duration cannot exceed "
            f"{QWEN_MAX_ALIGNMENT_DURATION_SECONDS:g} seconds, the forced-aligner limit"
        )


def validate_voxtral_delay(delay_ms: int) -> None:
    """Reject delays outside Voxtral's documented streaming presets."""
    if delay_ms < VOXTRAL_MIN_DELAY_MS or delay_ms > VOXTRAL_MAX_DELAY_MS:
        raise ValueError(
            "Voxtral delay must be between "
            f"{VOXTRAL_MIN_DELAY_MS}ms and {VOXTRAL_MAX_DELAY_MS}ms"
        )
    if (
        (delay_ms > 1200 and delay_ms != VOXTRAL_MAX_DELAY_MS)
        or delay_ms % VOXTRAL_DELAY_STEP_MS != 0
    ):
        raise ValueError(
            "Voxtral delay must be a multiple of "
            f"{VOXTRAL_DELAY_STEP_MS}ms up to 1200ms, or {VOXTRAL_MAX_DELAY_MS}ms"
        )


def validate_voxtral_timestamp_offset(offset_tokens: int) -> None:
    """Reject timestamp offsets outside Voxtral's 30-token delay horizon."""
    if not 0 <= offset_tokens <= VOXTRAL_MAX_TIMESTAMP_OFFSET_TOKENS:
        raise ValueError(
            "Voxtral timestamp offset must be between 0 and "
            f"{VOXTRAL_MAX_TIMESTAMP_OFFSET_TOKENS} tokens"
        )


@dataclass(frozen=True)
class PreparationConfig:
    """Settings owned by the prepare worker and its pyannote diarization pass."""

    input_path: Path
    output_directory: Path
    working_directory: Path
    device: str = "auto"
    pyannote_model: str = DEFAULT_PYANNOTE_MODEL
    language: str = DEFAULT_LANGUAGE
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    keep_intermediate_files: bool = False
    log_level: str = DEFAULT_LOG_LEVEL

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if not self.language:
            raise ValueError("language must not be empty")
        if self.num_speakers is not None and self.num_speakers < 1:
            raise ValueError("num_speakers must be positive")
        if self.min_speakers is not None and self.min_speakers < 1:
            raise ValueError("min_speakers must be positive")
        if self.max_speakers is not None and self.max_speakers < 1:
            raise ValueError("max_speakers must be positive")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot exceed max_speakers")

    @classmethod
    def from_environment(
        cls,
        input_path: Path,
        output_directory: Path,
        overrides: Mapping[str, object],
        env: Mapping[str, str] | None = None,
    ) -> PreparationConfig:
        """Create preparation configuration from CLI values, environment, then defaults."""
        choices = _Choices(overrides, os.environ if env is None else env)
        return cls(
            input_path=input_path,
            output_directory=output_directory,
            working_directory=_working_directory(overrides, choices.env),
            device=choices.string("device", "DEVICE", "auto"),
            pyannote_model=choices.string(
                "pyannote_model", "PYANNOTE_MODEL", DEFAULT_PYANNOTE_MODEL
            ),
            language=choices.string("language", "LANGUAGE", DEFAULT_LANGUAGE),
            num_speakers=choices.int_or_none("num_speakers", "NUM_SPEAKERS"),
            min_speakers=choices.int_or_none("min_speakers", "MIN_SPEAKERS"),
            max_speakers=choices.int_or_none("max_speakers", "MAX_SPEAKERS"),
            keep_intermediate_files=bool(
                overrides.get("keep_intermediate_files")
                or choices.env.get("KEEP_INTERMEDIATE_FILES") == "1"
            ),
            log_level=choices.string("log_level", "LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )


@dataclass(frozen=True)
class RecognitionConfig:
    """Settings owned by the recognize worker and the selected ASR backend.

    The recording language is not parsed from the environment: the prepared
    artifact owns it, and only an explicit ``--language`` flag may override it.
    """

    prepared_path: Path
    output_directory: Path
    working_directory: Path
    device: str = "auto"
    asr_backend: str = "parakeet"
    asr_model: str | None = None
    language: str | None = None

    qwen_aligner_model: str = DEFAULT_QWEN_ALIGNER_MODEL

    parakeet_segment_duration: float = DEFAULT_PARAKEET_SEGMENT_DURATION
    parakeet_segment_overlap: float = DEFAULT_PARAKEET_SEGMENT_OVERLAP

    qwen_segment_duration: float = DEFAULT_QWEN_SEGMENT_DURATION
    qwen_segment_overlap: float = DEFAULT_QWEN_SEGMENT_OVERLAP

    nemotron_num_lookahead_tokens: int | None = DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS

    voxtral_delay_ms: int = DEFAULT_VOXTRAL_DELAY_MS
    voxtral_timestamp_offset_tokens: int = DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS

    faster_whisper_compute_type: str = DEFAULT_FASTER_WHISPER_COMPUTE_TYPE

    canary_chunk_duration_seconds: float = DEFAULT_CANARY_CHUNK_DURATION_SECONDS

    log_level: str = DEFAULT_LOG_LEVEL

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.asr_backend not in ASR_BACKENDS:
            raise ValueError(f"asr_backend must be one of: {', '.join(ASR_BACKENDS)}")
        if self.language is not None and not self.language:
            raise ValueError("language must not be empty")
        for name, duration, overlap in (
            ("parakeet", self.parakeet_segment_duration, self.parakeet_segment_overlap),
            ("qwen", self.qwen_segment_duration, self.qwen_segment_overlap),
        ):
            if duration <= 0 or not 0 <= overlap < duration:
                raise ValueError(
                    f"{name} segment overlap must be non-negative and shorter than its duration"
                )
        validate_qwen_segment_duration(self.qwen_segment_duration)
        validate_voxtral_delay(self.voxtral_delay_ms)
        validate_voxtral_timestamp_offset(self.voxtral_timestamp_offset_tokens)
        if self.faster_whisper_compute_type not in FASTER_WHISPER_COMPUTE_TYPES:
            raise ValueError(
                "faster_whisper_compute_type must be one of: "
                + ", ".join(FASTER_WHISPER_COMPUTE_TYPES)
            )
        if self.canary_chunk_duration_seconds <= 0:
            raise ValueError("canary_chunk_duration_seconds must be positive")

    @property
    def resolved_asr_model(self) -> str:
        """Return an explicit ASR model or the selected backend's default model."""
        if self.asr_model:
            return self.asr_model
        return DEFAULT_ASR_MODELS[self.asr_backend]

    def language_for(self, prepared_language: str | None) -> str | None:
        """Inherit the prepared artifact's language unless explicitly overridden."""
        if self.language is not None:
            return self.language
        return prepared_language

    @classmethod
    def from_environment(
        cls,
        prepared_path: Path,
        output_directory: Path,
        overrides: Mapping[str, object],
        env: Mapping[str, str] | None = None,
    ) -> RecognitionConfig:
        """Create recognition configuration from CLI values, environment, then defaults."""
        choices = _Choices(overrides, os.environ if env is None else env)
        return cls(
            prepared_path=prepared_path,
            output_directory=output_directory,
            working_directory=_working_directory(overrides, choices.env),
            device=choices.string("device", "DEVICE", "auto"),
            asr_backend=choices.string("asr_backend", "ASR_BACKEND", "parakeet"),
            asr_model=choices.string_or_none("asr_model", "ASR_MODEL"),
            # Language has no environment variable; the prepared artifact owns it.
            language=choices.string_or_none("language", None),
            qwen_aligner_model=choices.string(
                "qwen_aligner_model", "QWEN_ALIGNER_MODEL", DEFAULT_QWEN_ALIGNER_MODEL
            ),
            parakeet_segment_duration=choices.float_or_default(
                "parakeet_segment_duration",
                "PARAKEET_SEGMENT_DURATION",
                DEFAULT_PARAKEET_SEGMENT_DURATION,
            ),
            parakeet_segment_overlap=choices.float_or_default(
                "parakeet_segment_overlap",
                "PARAKEET_SEGMENT_OVERLAP",
                DEFAULT_PARAKEET_SEGMENT_OVERLAP,
            ),
            qwen_segment_duration=choices.float_or_default(
                "qwen_segment_duration", "QWEN_SEGMENT_DURATION", DEFAULT_QWEN_SEGMENT_DURATION
            ),
            qwen_segment_overlap=choices.float_or_default(
                "qwen_segment_overlap", "QWEN_SEGMENT_OVERLAP", DEFAULT_QWEN_SEGMENT_OVERLAP
            ),
            nemotron_num_lookahead_tokens=choices.int_or_default(
                "nemotron_num_lookahead_tokens",
                "NEMOTRON_NUM_LOOKAHEAD_TOKENS",
                DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS,
            ),
            voxtral_delay_ms=choices.int_or_default(
                "voxtral_delay_ms", "VOXTRAL_DELAY_MS", DEFAULT_VOXTRAL_DELAY_MS
            ),
            voxtral_timestamp_offset_tokens=choices.int_or_default(
                "voxtral_timestamp_offset_tokens",
                "VOXTRAL_TIMESTAMP_OFFSET_TOKENS",
                DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS,
            ),
            faster_whisper_compute_type=choices.string(
                "faster_whisper_compute_type",
                "FASTER_WHISPER_COMPUTE_TYPE",
                DEFAULT_FASTER_WHISPER_COMPUTE_TYPE,
            ),
            canary_chunk_duration_seconds=choices.float_or_default(
                "canary_chunk_duration_seconds",
                "CANARY_CHUNK_DURATION",
                DEFAULT_CANARY_CHUNK_DURATION_SECONDS,
            ),
            log_level=choices.string("log_level", "LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )


@dataclass(frozen=True)
class FinalizationConfig:
    """Settings owned by the backend-neutral, CPU-only finalize worker."""

    output_directory: Path
    alignment_tolerance: float = DEFAULT_ALIGNMENT_TOLERANCE
    turn_gap_seconds: float = DEFAULT_TURN_GAP_SECONDS
    keep_intermediate_files: bool = False
    log_level: str = DEFAULT_LOG_LEVEL

    @classmethod
    def from_environment(
        cls,
        output_directory: Path,
        overrides: Mapping[str, object],
        env: Mapping[str, str] | None = None,
    ) -> FinalizationConfig:
        """Create finalization configuration from CLI values, environment, then defaults.

        No ASR backend or diarization setting is parsed here, so invalid
        backend environment variables cannot break finalization.
        """
        choices = _Choices(overrides, os.environ if env is None else env)
        return cls(
            output_directory=output_directory,
            alignment_tolerance=choices.float_or_default(
                "alignment_tolerance", None, DEFAULT_ALIGNMENT_TOLERANCE
            ),
            turn_gap_seconds=choices.float_or_default(
                "turn_gap_seconds", None, DEFAULT_TURN_GAP_SECONDS
            ),
            keep_intermediate_files=bool(
                overrides.get("keep_intermediate_files")
                or choices.env.get("KEEP_INTERMEDIATE_FILES") == "1"
            ),
            log_level=choices.string("log_level", "LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )