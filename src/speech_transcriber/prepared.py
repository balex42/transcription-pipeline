"""Versioned filesystem artifacts for prepared recordings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from speech_transcriber.models import AudioMetadata, DiarizationSegment, NormalizedAudio

SCHEMA_VERSION = 2
NORMALIZED_AUDIO_FILE = "normalized.wav"
DIARIZATION_FILE = "diarization.json"
MANIFEST_FILE = "prepared.json"


@dataclass(frozen=True)
class PreparedRecording:
    """One normalization and diarization result reusable by recognition and finalization."""

    audio: NormalizedAudio
    diarization: list[DiarizationSegment]
    work_directory: Path
    normalized_audio_sha256: str
    diarization_model: str | None = None
    language: str | None = None
    cleanup_enabled: bool = True


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a portable artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_prepared_recording(prepared: PreparedRecording, destination: Path) -> None:
    """Persist one prepared recording as an artifact bundle.

    The destination must be absent or empty so a job never silently mixes a new
    bundle with artifacts from a prior run.
    """
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(
            f"prepared artifact destination must be an empty directory: {destination}"
        )
    if not prepared.audio.path.is_file():
        raise ValueError(f"normalized audio is missing: {prepared.audio.path}")
    if not prepared.diarization_model:
        raise ValueError("prepared recording is missing diarization model provenance")
    if not prepared.language:
        raise ValueError("prepared recording is missing language provenance")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        shutil.copyfile(prepared.audio.path, staging / NORMALIZED_AUDIO_FILE)
        _write_json(
            staging / DIARIZATION_FILE,
            [asdict(segment) for segment in prepared.diarization],
        )
        _write_json(
            staging / MANIFEST_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "audio": {
                    "source": Path(prepared.audio.metadata.source).name,
                    "duration_seconds": prepared.audio.metadata.duration_seconds,
                    "sample_rate": prepared.audio.metadata.sample_rate,
                    "channels": prepared.audio.metadata.channels,
                    "sample_width_bits": prepared.audio.metadata.sample_width_bits,
                    "file": NORMALIZED_AUDIO_FILE,
                    "sha256": prepared.normalized_audio_sha256,
                },
                "diarization": {
                    "file": DIARIZATION_FILE,
                    "model": prepared.diarization_model,
                },
                "language": prepared.language,
            },
        )
        load_prepared_recording(staging)
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_prepared_recording(directory: Path) -> PreparedRecording:
    """Load and validate a prepared artifact without inspecting audio or networking."""
    manifest_path = directory / MANIFEST_FILE
    manifest = _object(_load_json(manifest_path, "prepared manifest"), "prepared manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported prepared artifact schema version: {manifest.get('schema_version')!r}"
        )

    audio = _object(manifest.get("audio"), "audio")
    diarization_info = _object(manifest.get("diarization"), "diarization")
    normalized_name = _artifact_file(audio.get("file"), NORMALIZED_AUDIO_FILE, "audio.file")
    diarization_name = _artifact_file(
        diarization_info.get("file"), DIARIZATION_FILE, "diarization.file"
    )
    normalized_path = directory / normalized_name
    if not normalized_path.is_file():
        raise ValueError(f"prepared normalized audio is missing: {normalized_path}")
    expected_sha256 = _sha256(audio.get("sha256"), "audio.sha256")
    if sha256_file(normalized_path) != expected_sha256:
        raise ValueError("prepared normalized audio SHA-256 does not match its manifest")

    metadata = AudioMetadata(
        source=_relative_name(audio.get("source"), "audio.source"),
        duration_seconds=_positive_float(audio.get("duration_seconds"), "audio.duration_seconds"),
        sample_rate=_positive_int(audio.get("sample_rate"), "audio.sample_rate"),
        channels=_positive_int(audio.get("channels"), "audio.channels"),
        sample_width_bits=_positive_int(audio.get("sample_width_bits"), "audio.sample_width_bits"),
    )
    if (metadata.sample_rate, metadata.channels, metadata.sample_width_bits) != (16_000, 1, 16):
        raise ValueError("prepared audio metadata must describe 16 kHz mono 16-bit PCM WAV")

    diarization_data = _load_json(directory / diarization_name, "diarization")
    if not isinstance(diarization_data, list):
        raise ValueError("diarization must be a JSON array")
    diarization = [
        _diarization_segment(record, index) for index, record in enumerate(diarization_data)
    ]
    return PreparedRecording(
        audio=NormalizedAudio(normalized_path, metadata),
        diarization=diarization,
        work_directory=directory,
        normalized_audio_sha256=expected_sha256,
        diarization_model=_string(diarization_info.get("model"), "diarization.model"),
        language=_string(manifest.get("language"), "language"),
        cleanup_enabled=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_json(path: Path, description: str) -> dict[str, Any] | list[Any]:
    if not path.is_file():
        raise ValueError(f"{description} is missing: {path}")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {path}") from error
    if not isinstance(value, dict | list):
        raise ValueError(f"{description} must be a JSON object or array")
    return value


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if Path(value).is_absolute():
        raise ValueError(f"{name} must not be an absolute path")
    return value


def _sha256(value: object, name: str) -> str:
    result = _string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _artifact_file(value: object, expected: str, name: str) -> str:
    result = _relative_name(value, name)
    if result != expected:
        raise ValueError(f"{name} must be the relative path {expected!r}")
    return result


def _relative_name(value: object, name: str) -> str:
    result = _string(value, name)
    if Path(result).is_absolute() or len(Path(result).parts) != 1:
        raise ValueError(f"{name} must be a relative filename")
    return result


def _diarization_segment(value: object, index: int) -> DiarizationSegment:
    record = _object(value, f"diarization[{index}]")
    speaker = _string(record.get("speaker"), f"diarization[{index}].speaker")
    start = _nonnegative_float(record.get("start"), f"diarization[{index}].start")
    end = _nonnegative_float(record.get("end"), f"diarization[{index}].end")
    if end < start:
        raise ValueError(f"diarization[{index}].end must not precede start")
    return DiarizationSegment(speaker=speaker, start=start, end=end)


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return result
