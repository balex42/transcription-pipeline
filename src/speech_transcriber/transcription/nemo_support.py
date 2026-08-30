"""Shared low-level NeMo runtime mechanics for checkpoint-based ASR backends.

This module owns only generic NeMo mechanics that several adapters reuse:

- strict offline resolution of the expected ``.nemo`` checkpoint
- lazy ``ASRModel.restore_from()`` restoration
- NeMo/PyTorch/CUDA runtime provenance
- safe extraction of ``Hypothesis.timestamp["word"]`` records

It deliberately knows nothing about long-form strategy: segmentation,
chunking, single-pass behavior, validation tolerances, and rebase rules stay
with each backend adapter. Importing this module never imports NeMo.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from speech_transcriber.errors import ModelLoadError
from speech_transcriber.model_cache import resolve_hf_snapshot
from speech_transcriber.models import RuntimeProvenance

LOGGER = logging.getLogger(__name__)


def resolve_checkpoint_path(
    model_reference: str, filename: str, *, subject: str = "model"
) -> tuple[str, str | None]:
    """Resolve one expected ``.nemo`` checkpoint without a runtime Hub lookup.

    A configured local file or directory takes precedence. Repository IDs are
    resolved from the active Hugging Face cache through the shared snapshot
    helper via ``refs/main``; with no usable ref exactly one cached snapshot is
    used deterministically. Resolution is always strict: runtime model
    downloading is intentionally unsupported, so a missing or ambiguous cache
    fails instead of returning a Hub repository ID; prefetch the repository
    before air-gapped use.

    Returns the resolved checkpoint path and, when it was located through the
    cache, the snapshot revision.
    """
    configured = Path(model_reference).expanduser()
    if configured.is_absolute() or configured.exists():
        return model_file(configured, filename, subject=subject), None

    snapshot = resolve_hf_snapshot(model_reference, subject=subject)
    return model_file(snapshot, filename, subject=subject), snapshot_revision(snapshot)


def model_file(path: Path, filename: str, *, subject: str) -> str:
    """Return the exact checkpoint file inside ``path``, failing strictly."""
    artifact = path / filename if path.is_dir() else path
    if artifact.is_file() and artifact.name == filename:
        return str(artifact)
    raise ModelLoadError(
        f"required {subject} artifact {filename!r} is missing from {path}"
    )


def snapshot_revision(snapshot: Path) -> str:
    """Return the cached snapshot revision, or ``unknown`` outside the cache."""
    if snapshot.parent.name == "snapshots" and snapshot.name:
        return snapshot.name
    return "unknown"


def restore_model(model_path: str, device: str, *, subject: str) -> Any:
    """Restore one trusted ``.nemo`` checkpoint, importing NeMo lazily."""
    from nemo.collections.asr.models import ASRModel

    return ASRModel.restore_from(model_path).to(device).eval()


def package_version(package: str) -> str:
    """Return an installed package version without importing it."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def torch_cuda_version() -> str:
    """Return the runtime CUDA version reported by PyTorch, when present."""
    try:
        import torch
    except ImportError:
        return "unknown"
    cuda = getattr(torch.version, "cuda", None)
    return cuda if isinstance(cuda, str) else "unknown"


def nemo_runtime_provenance() -> RuntimeProvenance:
    """Collect NeMo/PyTorch/CUDA provenance after a checkpoint is restored."""
    return RuntimeProvenance(
        name="nemo",
        version=package_version("nemo-toolkit"),
        components={
            "torch": package_version("torch"),
            "cuda": torch_cuda_version(),
        },
    )


def initial_runtime_provenance() -> RuntimeProvenance:
    """Return the pre-load provenance placeholder shared by NeMo adapters."""
    return RuntimeProvenance(
        name="nemo",
        version="unknown",
        components={"torch": "unknown", "cuda": "unknown"},
    )


def word_timestamp_records(hypothesis: Any, *, subject: str) -> Sequence[Mapping[str, Any]]:
    """Extract ``Hypothesis.timestamp['word']`` without assuming its type.

    Validates that the hypothesis carries mapping timestamp metadata whose
    ``word`` entry is a non-string sequence; the caller owns per-record and
    ordering semantics.
    """
    timestamp = getattr(hypothesis, "timestamp", None)
    if not isinstance(timestamp, Mapping):
        raise ValueError(f"{subject} output is missing timestamp metadata")
    records = timestamp.get("word")
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError(f"{subject} output is missing word timestamps")
    return records