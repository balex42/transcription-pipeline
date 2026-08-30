"""Configuration loading with CLI, environment, then default precedence."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
DEFAULT_CANARY_MODEL = "nvidia/canary-1b-v2"
DEFAULT_CANARY_CHUNK_DURATION_SECONDS = 10.0
DEFAULT_PRIMELINE_MODEL = "primeline/parakeet-primeline"
PRIMELINE_MODEL_FILE = "2_95_WER.nemo"
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
ASR_BACKENDS = ("parakeet", "primeline", "qwen", "nemotron", "voxtral", "faster-whisper", "canary")
COMPARE_BACKENDS = ("parakeet", "primeline", "qwen", "nemotron", "voxtral")
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


def _env_int(env: Mapping[str, str], name: str) -> int | None:
    value = env.get(name)
    return int(value) if value else None


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = env.get(name)
    return float(value) if value else default


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for one local or batch transcription job."""

    input_path: Path
    output_directory: Path
    working_directory: Path
    device: str = "auto"
    asr_backend: str = "parakeet"
    asr_model: str | None = None
    qwen_aligner_model: str = DEFAULT_QWEN_ALIGNER_MODEL
    pyannote_model: str = DEFAULT_PYANNOTE_MODEL
    parakeet_segment_duration: float = DEFAULT_PARAKEET_SEGMENT_DURATION
    parakeet_segment_overlap: float = DEFAULT_PARAKEET_SEGMENT_OVERLAP
    qwen_segment_duration: float = DEFAULT_QWEN_SEGMENT_DURATION
    qwen_segment_overlap: float = DEFAULT_QWEN_SEGMENT_OVERLAP
    nemotron_num_lookahead_tokens: int | None = DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS
    voxtral_delay_ms: int = DEFAULT_VOXTRAL_DELAY_MS
    voxtral_timestamp_offset_tokens: int = DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS
    faster_whisper_compute_type: str = DEFAULT_FASTER_WHISPER_COMPUTE_TYPE
    canary_chunk_duration_seconds: float = DEFAULT_CANARY_CHUNK_DURATION_SECONDS
    language: str = "de-DE"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    keep_intermediate_files: bool = False
    log_level: str = "INFO"
    alignment_tolerance: float = 0.25
    turn_gap_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.asr_backend not in ASR_BACKENDS:
            raise ValueError(f"asr_backend must be one of: {', '.join(ASR_BACKENDS)}")
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

    @property
    def resolved_asr_model(self) -> str:
        """Return an explicit ASR model or the selected backend's default model."""
        if self.asr_model:
            return self.asr_model
        return DEFAULT_ASR_MODELS[self.asr_backend]

    @classmethod
    def from_environment(
        cls,
        input_path: Path,
        output_directory: Path,
        overrides: Mapping[str, object],
        env: Mapping[str, str] | None = None,
    ) -> PipelineConfig:
        """Create configuration using explicit values before environment values."""
        values = os.environ if env is None else env
        if overrides.get("granite_model") is not None or "GRANITE_MODEL" in values:
            raise ValueError("GRANITE_MODEL is no longer supported; use ASR_BACKEND=qwen instead")

        def choose(name: str, env_name: str, default: str) -> str:
            value = overrides.get(name)
            return str(value) if value is not None else values.get(env_name, default)

        def choose_float(name: str, env_name: str, default: float) -> float:
            value = overrides.get(name)
            if value is None:
                return _env_float(values, env_name, default)
            if not isinstance(value, str | int | float):
                raise ValueError(f"{name} must be numeric")
            return float(value)

        def choose_int(name: str, env_name: str) -> int | None:
            value = overrides.get(name)
            if value is None:
                return _env_int(values, env_name)
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            return value

        asr_backend = choose("asr_backend", "ASR_BACKEND", "parakeet")
        nemotron_lookahead = choose_int(
            "nemotron_num_lookahead_tokens", "NEMOTRON_NUM_LOOKAHEAD_TOKENS"
        )
        voxtral_delay = choose_int("voxtral_delay_ms", "VOXTRAL_DELAY_MS")
        voxtral_timestamp_offset = choose_int(
            "voxtral_timestamp_offset_tokens", "VOXTRAL_TIMESTAMP_OFFSET_TOKENS"
        )
        faster_whisper_compute_type = choose(
            "faster_whisper_compute_type",
            "FASTER_WHISPER_COMPUTE_TYPE",
            DEFAULT_FASTER_WHISPER_COMPUTE_TYPE,
        )
        return cls(
            input_path=input_path,
            output_directory=output_directory,
            working_directory=Path(
                choose("working_directory", "WORKING_DIRECTORY", str(Path.cwd() / "work"))
            ),
            device=choose("device", "DEVICE", "auto"),
            asr_backend=asr_backend,
            asr_model=(
                str(overrides["asr_model"])
                if overrides.get("asr_model") is not None
                else values.get("ASR_MODEL")
            ),
            qwen_aligner_model=choose(
                "qwen_aligner_model", "QWEN_ALIGNER_MODEL", DEFAULT_QWEN_ALIGNER_MODEL
            ),
            pyannote_model=choose("pyannote_model", "PYANNOTE_MODEL", DEFAULT_PYANNOTE_MODEL),
            parakeet_segment_duration=choose_float(
                "parakeet_segment_duration",
                "PARAKEET_SEGMENT_DURATION",
                DEFAULT_PARAKEET_SEGMENT_DURATION,
            ),
            parakeet_segment_overlap=choose_float(
                "parakeet_segment_overlap",
                "PARAKEET_SEGMENT_OVERLAP",
                DEFAULT_PARAKEET_SEGMENT_OVERLAP,
            ),
            qwen_segment_duration=choose_float(
                "qwen_segment_duration",
                "QWEN_SEGMENT_DURATION",
                DEFAULT_QWEN_SEGMENT_DURATION,
            ),
            qwen_segment_overlap=choose_float(
                "qwen_segment_overlap",
                "QWEN_SEGMENT_OVERLAP",
                DEFAULT_QWEN_SEGMENT_OVERLAP,
            ),
            nemotron_num_lookahead_tokens=(
                DEFAULT_NEMOTRON_NUM_LOOKAHEAD_TOKENS
                if nemotron_lookahead is None
                else nemotron_lookahead
            ),
            voxtral_delay_ms=(
                DEFAULT_VOXTRAL_DELAY_MS if voxtral_delay is None else voxtral_delay
            ),
            voxtral_timestamp_offset_tokens=(
                DEFAULT_VOXTRAL_TIMESTAMP_OFFSET_TOKENS
                if voxtral_timestamp_offset is None
                else voxtral_timestamp_offset
            ),
            faster_whisper_compute_type=faster_whisper_compute_type,
            canary_chunk_duration_seconds=choose_float(
                "canary_chunk_duration_seconds",
                "CANARY_CHUNK_DURATION",
                DEFAULT_CANARY_CHUNK_DURATION_SECONDS,
            ),
            language=choose("language", "LANGUAGE", "de-DE"),
            num_speakers=choose_int("num_speakers", "NUM_SPEAKERS"),
            min_speakers=choose_int("min_speakers", "MIN_SPEAKERS"),
            max_speakers=choose_int("max_speakers", "MAX_SPEAKERS"),
            keep_intermediate_files=bool(
                overrides.get("keep_intermediate_files")
                or values.get("KEEP_INTERMEDIATE_FILES") == "1"
            ),
            log_level=choose("log_level", "LOG_LEVEL", "INFO").upper(),
        )
