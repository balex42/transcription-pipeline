"""Versioned filesystem artifacts exchanged between ASR and transcript finalization."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from speech_transcriber.models import (
    ASRRecognitionResult,
    ASRRunMetadata,
    ASRWord,
    RuntimeProvenance,
)

SCHEMA_VERSION = 1
ASR_WORDS_FILE = "asr_words.json"
METADATA_FILE = "metadata.json"


def write_asr_recognition(result: ASRRecognitionResult, destination: Path) -> None:
    """Write a relocatable recognition artifact without backend-private state."""
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"ASR artifact destination must be an empty directory: {destination}")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        write_asr_result_files(result, staging)
        load_asr_recognition(staging)
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_asr_result_files(result: ASRRecognitionResult, directory: Path) -> None:
    """Write the two portable ASR files into an existing final result directory."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / ASR_WORDS_FILE, [asdict(word) for word in result.words])
    _write_json(directory / METADATA_FILE, _metadata_payload(result.metadata))


def load_asr_recognition(directory: Path) -> ASRRecognitionResult:
    """Load a recognition artifact without importing or instantiating an ASR backend."""
    metadata = _object(_load_json(directory / METADATA_FILE, "ASR metadata"), "ASR metadata")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ASR artifact schema version: {metadata.get('schema_version')!r}"
        )
    if metadata.get("asr_words_file") != ASR_WORDS_FILE:
        raise ValueError(f"asr_words_file must be the relative path {ASR_WORDS_FILE!r}")
    word_data = _load_json(directory / ASR_WORDS_FILE, "ASR words")
    if not isinstance(word_data, list):
        raise ValueError("ASR words must be a JSON array")
    return ASRRecognitionResult(
        words=[_asr_word(record, index) for index, record in enumerate(word_data)],
        metadata=_asr_metadata(metadata),
    )


def _metadata_payload(metadata: ASRRunMetadata) -> dict[str, object]:
    payload = _remove_absolute_paths(asdict(metadata))
    assert isinstance(payload, dict)
    return {
        "schema_version": SCHEMA_VERSION,
        "asr_words_file": ASR_WORDS_FILE,
        **payload,
    }


def _remove_absolute_paths(value: object) -> object:
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name or "redacted-path"
    if isinstance(value, dict):
        return {str(key): _remove_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_remove_absolute_paths(item) for item in value]
    return value


def _asr_metadata(value: dict[str, Any]) -> ASRRunMetadata:
    runtime = _object(value.get("runtime"), "runtime")
    return ASRRunMetadata(
        backend=_string(value.get("backend"), "backend"),
        model=_string(value.get("model"), "model"),
        device=_string(value.get("device"), "device"),
        dtype=_string(value.get("dtype"), "dtype"),
        audio_duration_seconds=_positive_float(
            value.get("audio_duration_seconds"), "audio_duration_seconds"
        ),
        model_load_seconds=_nonnegative_float(
            value.get("model_load_seconds"), "model_load_seconds"
        ),
        transcription_seconds=_nonnegative_float(
            value.get("transcription_seconds"), "transcription_seconds"
        ),
        total_asr_seconds=_nonnegative_float(value.get("total_asr_seconds"), "total_asr_seconds"),
        real_time_factor=_nonnegative_float(value.get("real_time_factor"), "real_time_factor"),
        peak_cuda_memory_allocated_bytes=_optional_nonnegative_int(
            value.get("peak_cuda_memory_allocated_bytes"), "peak_cuda_memory_allocated_bytes"
        ),
        peak_cuda_memory_reserved_bytes=_optional_nonnegative_int(
            value.get("peak_cuda_memory_reserved_bytes"), "peak_cuda_memory_reserved_bytes"
        ),
        transformers_version=_string_or_default(
            value.get("transformers_version"), "transformers_version", "unknown"
        ),
        torch_version=_string_or_default(value.get("torch_version"), "torch_version", "unknown"),
        runtime=RuntimeProvenance(
            name=_string(runtime.get("name"), "runtime.name"),
            version=_string(runtime.get("version"), "runtime.version"),
            components=_string_mapping_or_empty(runtime.get("components"), "runtime.components"),
        ),
        backend_metrics=_number_mapping_or_empty(value.get("backend_metrics"), "backend_metrics"),
        backend_models=_string_mapping_or_empty(value.get("backend_models"), "backend_models"),
        backend_configuration=_configuration_or_empty(value.get("backend_configuration")),
    )


def _asr_word(value: object, index: int) -> ASRWord:
    record = _object(value, f"ASR word {index}")
    unexpected = set(record) - {"text", "start", "end", "confidence"}
    if unexpected:
        raise ValueError(f"ASR word {index} has unsupported fields: {sorted(unexpected)!r}")
    start = _optional_nonnegative_float(record.get("start"), f"ASR word {index}.start")
    end = _nonnegative_float(record.get("end"), f"ASR word {index}.end")
    if start is not None and end < start:
        raise ValueError(f"ASR word {index}.end must not precede start")
    return ASRWord(
        text=_string(record.get("text"), f"ASR word {index}.text"),
        start=start,
        end=end,
        confidence=_optional_finite_float(record.get("confidence"), f"ASR word {index}.confidence"),
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


def _string_or_default(value: object, name: str, default: str) -> str:
    return default if value is None else _string(value, name)


def _positive_float(value: object, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _optional_nonnegative_float(value: object, name: str) -> float | None:
    return None if value is None else _nonnegative_float(value, name)


def _optional_finite_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number or null")
    return float(value)


def _optional_nonnegative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")
    return value


def _string_mapping(value: object, name: str) -> dict[str, str]:
    mapping = _object(value, name)
    return {key: _string(item, f"{name}.{key}") for key, item in mapping.items()}


def _string_mapping_or_empty(value: object, name: str) -> dict[str, str]:
    return {} if value is None else _string_mapping(value, name)


def _number_mapping(value: object, name: str) -> dict[str, float]:
    mapping = _object(value, name)
    return {key: _nonnegative_float(item, f"{name}.{key}") for key, item in mapping.items()}


def _number_mapping_or_empty(value: object, name: str) -> dict[str, float]:
    return {} if value is None else _number_mapping(value, name)


def _configuration(value: object) -> dict[str, str | int | float | bool | None]:
    mapping = _object(value, "backend_configuration")
    result: dict[str, str | int | float | bool | None] = {}
    for key, item in mapping.items():
        if (
            item is None
            or isinstance(item, str | bool)
            or (
                isinstance(item, int | float)
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            )
        ):
            result[key] = item
        else:
            raise ValueError(f"backend_configuration.{key} must be a JSON scalar")
    return result


def _configuration_or_empty(value: object) -> dict[str, str | int | float | bool | None]:
    return {} if value is None else _configuration(value)
