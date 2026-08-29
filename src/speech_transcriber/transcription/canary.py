"""NVIDIA NeMo Canary 1B v2 adapter with native word timestamps.

The adapter restores the trusted ``.nemo`` checkpoint from an explicit local
path or a resolved Hugging Face cache snapshot. It deliberately never calls
``from_pretrained()`` during recognition, so air-gapped runs cannot fall back
to an online model lookup.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.language import normalize_language
from speech_transcriber.models import ASRWord, NormalizedAudio, RuntimeProvenance
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities

CANARY_MODEL_FILE = "canary-1b-v2.nemo"
SUPPORTED_CANARY_LANGUAGES = frozenset(
    {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    }
)


class CanaryTranscriber(Transcriber):
    """Transcribe a complete normalized recording with NeMo Canary 1B v2."""

    capabilities = TranscriberCapabilities(True, True, True, True)

    def __init__(self, model: str, device: str, language: str) -> None:
        self.model_reference = model
        self.device = device
        self.requested_language = language
        self.source_language = canary_language(language)
        self.target_language = self.source_language
        # The checkpoint controls its precision; do not infer one from the
        # process-wide PyTorch default or force a conversion.
        self.dtype_name = "checkpoint-default"
        self._model: Any | None = None
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        self.backend_configuration: dict[str, str | int | float | bool | None] = {
            "requested_language": language,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "timestamps": True,
            "batch_size": 1,
            "long_form_mode": "native_dynamic_chunking",
        }
        self.runtime_provenance = RuntimeProvenance(
            name="nemo",
            version="unknown",
            components={"torch": "unknown", "cuda": "unknown"},
        )

    def load(self) -> None:
        """Restore the local NeMo checkpoint without contacting Hugging Face."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Return NeMo's absolute native word timestamps as canonical words."""
        model = self._load()
        try:
            # A one-file, batch-size-one request enables NeMo's supported
            # dynamic long-form chunking for Canary when the checkpoint embeds
            # its timestamp alignment model.
            outputs = model.transcribe(
                [str(audio.path)],
                batch_size=1,
                return_hypotheses=True,
                source_lang=self.source_language,
                target_lang=self.target_language,
                timestamps=True,
            )
            words = flatten_canary_words(outputs)
            self.backend_metrics["word_count"] = float(len(words))
            return words
        except (ASROutputError, ModelLoadError):
            raise
        except Exception as error:
            raise ASROutputError(
                "Canary recognition failed for "
                f"{audio.path} with model {self.model_reference}: {error}"
            ) from error

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        model_path = resolve_canary_model_path(self.model_reference)
        try:
            self._model = _restore_canary_model(model_path, self.device)
        except Exception as error:
            raise ModelLoadError(
                f"could not load Canary model {self.model_reference}: {error}"
            ) from error
        self.backend_models["model_file"] = model_path
        self.runtime_provenance = RuntimeProvenance(
            name="nemo",
            version=_package_version("nemo-toolkit"),
            components={
                "torch": _package_version("torch"),
                "cuda": _torch_cuda_version(),
            },
        )
        return self._model

    def release(self) -> None:
        """Drop the NeMo model so the runner can release GPU memory between jobs."""
        self._model = None


def canary_language(language: str | None) -> str:
    """Normalize and validate the explicit Canary source/target language."""
    normalized = normalize_language(language)
    if normalized not in SUPPORTED_CANARY_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_CANARY_LANGUAGES))
        raise ValueError(
            f"Canary does not support language {language!r}; use one of: {supported}"
        )
    return normalized


def resolve_canary_model_path(model: str) -> str:
    """Find Canary's ``.nemo`` checkpoint without a runtime Hub lookup.

    A configured local file or directory takes precedence. Repository IDs are
    resolved from the active Hugging Face cache via ``refs/main`` so multiple
    historical snapshots remain deterministic. Runtime model downloading is
    intentionally unsupported; prefetch the repository before air-gapped use.
    """
    configured = Path(model).expanduser()
    if configured.is_absolute() or configured.exists():
        return _canary_model_file(configured)

    cache_root = os.environ.get("HF_HOME")
    repo_dir = (
        Path(cache_root) / "hub" / f"models--{model.replace('/', '--')}"
        if cache_root
        else None
    )
    if repo_dir is None or not repo_dir.is_dir():
        location = str(repo_dir) if repo_dir is not None else "the configured HF_HOME cache"
        raise ModelLoadError(
            f"required Canary model {model!r} is not present in the offline model cache "
            f"under {location}"
        )

    snapshots = repo_dir / "snapshots"
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        if revision and Path(revision).name == revision:
            snapshot = snapshots / revision
            if snapshot.is_dir():
                return _canary_model_file(snapshot)

    revisions = (
        sorted(path for path in snapshots.iterdir() if path.is_dir()) if snapshots.is_dir() else []
    )
    if len(revisions) == 1:
        return _canary_model_file(revisions[0])
    if not revisions:
        raise ModelLoadError(
            f"required Canary model {model!r} is not present in the offline model cache "
            f"under {repo_dir}"
        )
    raise ModelLoadError(
        f"required Canary model {model!r} has no resolvable cached snapshot under {repo_dir}"
    )


def flatten_canary_words(outputs: Sequence[Any]) -> list[ASRWord]:
    """Map ``Hypothesis.timestamp['word']`` records to canonical ASR words."""
    if len(outputs) != 1:
        raise ASROutputError(f"Canary returned {len(outputs)} hypotheses for one recording")
    timestamp = getattr(outputs[0], "timestamp", None)
    if not isinstance(timestamp, Mapping):
        raise ASROutputError("Canary output is missing timestamp metadata")
    records = timestamp.get("word")
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ASROutputError("Canary output is missing word timestamps")

    words: list[ASRWord] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ASROutputError("Canary word timestamp must be an object")
        text = record.get("word")
        start = record.get("start")
        end = record.get("end")
        if not isinstance(text, str) or not text.strip():
            raise ASROutputError("Canary word timestamp is missing text")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ASROutputError("Canary word timestamp is missing numeric start/end values")
        if end < start:
            raise ASROutputError("Canary word timestamp ends before it starts")
        # NeMo's Canary timestamp records expose no stable per-word confidence.
        words.append(
            ASRWord(text=text.strip(), start=float(start), end=float(end), confidence=None)
        )
    return words


def _canary_model_file(path: Path) -> str:
    artifact = path / CANARY_MODEL_FILE if path.is_dir() else path
    if artifact.is_file() and artifact.name.endswith(".nemo"):
        return str(artifact)
    raise ModelLoadError(
        f"required Canary model artifact {CANARY_MODEL_FILE!r} is missing from {path}"
    )


def _restore_canary_model(model_path: str, device: str) -> Any:
    """Restore and place Canary on the requested device, importing NeMo lazily."""
    from nemo.collections.asr.models import ASRModel

    return ASRModel.restore_from(model_path).to(device).eval()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _torch_cuda_version() -> str:
    try:
        import torch
    except ImportError:
        return "unknown"
    cuda = getattr(torch.version, "cuda", None)
    return cuda if isinstance(cuda, str) else "unknown"
