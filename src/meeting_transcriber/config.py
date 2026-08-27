"""Configuration loading with CLI, environment, then default precedence."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GRANITE_MODEL = "ibm-granite/granite-speech-4.1-2b-plus"
DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3"
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
ASR_BACKENDS = ("parakeet", "whisper", "granite")
DEFAULT_CHUNK_SETTINGS = {
    "parakeet": (180.0, 15.0),
    "whisper": (180.0, 15.0),
    "granite": (90.0, 10.0),
}
DEFAULT_ASR_MODELS = {
    "parakeet": DEFAULT_PARAKEET_MODEL,
    "whisper": DEFAULT_WHISPER_MODEL,
    "granite": DEFAULT_GRANITE_MODEL,
}
DEFAULT_TIMESTAMP_PROMPT = (
    "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag "
    "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
)
DEFAULT_SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
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
    # Retained as a Granite-only compatibility alias for existing deployments.
    granite_model: str = DEFAULT_GRANITE_MODEL
    pyannote_model: str = DEFAULT_PYANNOTE_MODEL
    chunk_duration: float = 180.0
    chunk_overlap: float = 15.0
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    keep_intermediate_files: bool = False
    log_level: str = "INFO"
    granite_timestamp_prompt: str = DEFAULT_TIMESTAMP_PROMPT
    granite_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 10_000
    alignment_tolerance: float = 0.25
    turn_gap_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.asr_backend not in ASR_BACKENDS:
            raise ValueError(f"asr_backend must be one of: {', '.join(ASR_BACKENDS)}")
        if self.chunk_duration <= 0 or not 0 <= self.chunk_overlap < self.chunk_duration:
            raise ValueError("chunk_overlap must be non-negative and shorter than chunk_duration")
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
        if self.asr_backend == "granite":
            return self.granite_model
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
        default_chunk_duration, default_chunk_overlap = DEFAULT_CHUNK_SETTINGS.get(
            asr_backend, DEFAULT_CHUNK_SETTINGS["parakeet"]
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
            granite_model=choose("granite_model", "GRANITE_MODEL", DEFAULT_GRANITE_MODEL),
            pyannote_model=choose("pyannote_model", "PYANNOTE_MODEL", DEFAULT_PYANNOTE_MODEL),
            chunk_duration=choose_float(
                "chunk_duration", "CHUNK_DURATION", default_chunk_duration
            ),
            chunk_overlap=choose_float("chunk_overlap", "CHUNK_OVERLAP", default_chunk_overlap),
            num_speakers=choose_int("num_speakers", "NUM_SPEAKERS"),
            min_speakers=choose_int("min_speakers", "MIN_SPEAKERS"),
            max_speakers=choose_int("max_speakers", "MAX_SPEAKERS"),
            keep_intermediate_files=bool(
                overrides.get("keep_intermediate_files")
                or values.get("KEEP_INTERMEDIATE_FILES") == "1"
            ),
            log_level=choose("log_level", "LOG_LEVEL", "INFO").upper(),
        )
