"""Neutral Hugging Face cache snapshot resolution for offline model loading.

This module knows only the shared Hugging Face hub cache layout:

- an absolute local filesystem path is returned untouched
- repository IDs map to ``$HF_HOME/hub/models--<org>--<name>/``
- ``refs/main`` selects the active ``snapshots/<revision>/`` directory
- exactly one snapshot is used when the ref is missing or unusable
- ambiguity and missing repositories fail instead of guessing online

It deliberately knows nothing about Voxtral, NeMo, CTranslate2,
Transformers, ASR, or model-specific artifact filenames; backends keep their
own artifact handling and decide how strict the offline resolution is.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from speech_transcriber.errors import ModelLoadError

LOGGER = logging.getLogger(__name__)


def resolve_hf_snapshot(
    model_reference: str,
    *,
    hf_home: str | None = None,
    offline: bool | None = None,
    fallback_to_reference: bool = False,
    subject: str = "model",
) -> Path:
    """Resolve a model reference to one deterministic local snapshot directory.

    Semantics:

    - an absolute local path is returned unchanged and is never reinterpreted
      as a Hub ID; existence is the caller's contract to enforce
    - a valid ``refs/main`` pointing at an existing snapshot is honored
    - with no usable ref, exactly one cached snapshot is used deterministically
    - multiple snapshots without a usable ref fail without guessing
    - a repository missing from the cache fails when offline

    ``offline`` defaults to ``HF_HUB_OFFLINE == "1"``. With
    ``fallback_to_reference=True`` a reference that cannot be resolved while
    online falls back to returning the repository ID instead of raising;
    offline resolution never falls back.

    ``subject`` only shapes error text, e.g. ``"Voxtral model"``.
    """
    if Path(model_reference).is_absolute():
        return Path(model_reference)

    offline = (os.environ.get("HF_HUB_OFFLINE") == "1") if offline is None else offline
    repo_dir = hub_repository_dir(model_reference, hf_home=hf_home)
    if repo_dir is None or not repo_dir.is_dir():
        if fallback_to_reference and not offline:
            return Path(model_reference)
        raise ModelLoadError(
            f"required {subject} {model_reference!r} is not present in the offline model "
            "cache; set HF_HOME to the cache root or configure a local model path"
        )

    snapshot = _selected_snapshot(repo_dir, model_reference)
    if snapshot is not None:
        return snapshot
    if fallback_to_reference and not offline:
        return Path(model_reference)
    if _sorted_revisions(repo_dir / "snapshots"):
        raise ModelLoadError(
            f"required {subject} {model_reference!r} has no resolvable cached snapshot "
            f"under {repo_dir}; refusing to guess between multiple cached snapshots"
        )
    raise ModelLoadError(
        f"required {subject} {model_reference!r} is not present in the offline model "
        f"cache under {repo_dir}"
    )


def hub_repository_dir(model_reference: str, *, hf_home: str | None = None) -> Path | None:
    """Return the cached hub repository directory for a model ID, if configured.

    ``None`` means no ``HF_HOME``-rooted cache can apply to this reference;
    it never means the repository is present.
    """
    root = hf_home if hf_home is not None else os.environ.get("HF_HOME")
    if not root:
        return None
    return Path(root) / "hub" / f"models--{model_reference.replace('/', '--')}"


def snapshot_revision(snapshot: Path) -> str | None:
    """Return a snapshot directory's revision name for concise logging."""
    name = Path(snapshot).name
    return name or None


def _selected_snapshot(repo_dir: Path, model_reference: str) -> Path | None:
    """Pick one snapshot inside an existing repository directory.

    ``refs/main`` wins when it names an existing snapshot. Otherwise exactly
    one cached snapshot is used deterministically; anything else yields
    ``None`` so the caller can raise or fall back.
    """
    revision = _main_revision(repo_dir)
    if revision is not None:
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.is_dir():
            LOGGER.info("resolved %s via cached refs/main", model_reference)
            return snapshot
        LOGGER.warning(
            "refs/main for %s points at a missing snapshot revision; falling back to "
            "the unique cached snapshot when exactly one exists",
            model_reference,
        )

    revisions = _sorted_revisions(repo_dir / "snapshots")
    if len(revisions) == 1:
        LOGGER.info("resolved %s to the only cached snapshot", model_reference)
        return revisions[0]
    return None


def _main_revision(repo_dir: Path) -> str | None:
    """Return a safe ``refs/main`` revision name, or ``None`` when unusable."""
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        return None
    revision = ref.read_text(encoding="utf-8").strip()
    if revision and Path(revision).name == revision:
        return revision
    return None


def _sorted_revisions(snapshots_dir: Path) -> list[Path]:
    if not snapshots_dir.is_dir():
        return []
    return sorted(path for path in snapshots_dir.iterdir() if path.is_dir())