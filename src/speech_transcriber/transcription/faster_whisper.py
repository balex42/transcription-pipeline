"""CTranslate2 faster-whisper adapter for Systran Whisper large-v3.

This backend is a distinct heterogeneous runtime: it uses the native
``faster-whisper`` / CTranslate2 stack and never imports Transformers for
inference. It is not a restoration of the removed Transformers Whisper
backend.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, cast

from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.language import normalize_language
from speech_transcriber.models import ASRWord, NormalizedAudio, RuntimeProvenance
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities

DEFAULT_BEAM_SIZE = 5


class _FasterWhisperInfo(Protocol):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float


class _FasterWhisperModel(Protocol):
    def transcribe(
        self,
        audio: str | Path,
        *,
        language: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        word_timestamps: bool = False,
        vad_filter: bool = False,
        **kwargs: object,
    ) -> tuple[Iterable[Any], _FasterWhisperInfo]: ...


class FasterWhisperTranscriber(Transcriber):
    """Transcribe one normalized recording with native word timestamps."""

    capabilities = TranscriberCapabilities(True, True, True, True)

    def __init__(
        self,
        model: str,
        device: str,
        language: str | None = None,
        compute_type: str = "float16",
    ) -> None:
        self.model_reference = model
        self.device = device
        self.language = language
        self.compute_type = compute_type
        self.dtype_name = compute_type
        self._model: _FasterWhisperModel | None = None
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        self.backend_configuration: dict[str, str | int | float | bool | None] = {
            "language": language,
            "compute_type": compute_type,
            "word_timestamps": True,
            "vad_filter": False,
            "beam_size": DEFAULT_BEAM_SIZE,
        }
        self.runtime_provenance = RuntimeProvenance(
            name="faster-whisper",
            version="unknown",
            components={
                "ctranslate2": "unknown",
                "huggingface_hub": "unknown",
            },
        )

    def load(self) -> None:
        """Load the CTranslate2 Whisper model from a local cached snapshot."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Flatten faster-whisper segment words into canonical ASR words.

        The real faster-whisper API returns ``(segments, info)``; language
        metadata comes from ``info``, not from individual segments.
        """
        model = self._load()
        try:
            segments, info = model.transcribe(
                audio.path,
                language=whisper_language(self.language),
                beam_size=DEFAULT_BEAM_SIZE,
                word_timestamps=True,
                vad_filter=False,
            )
            words = flatten_segment_words(list(segments))
            self._record_info(info)
            return words
        except ASROutputError:
            raise
        except Exception as error:
            raise ASROutputError(
                "faster-whisper recognition failed for "
                f"{audio.path} with model {self.model_reference}: {error}"
            ) from error

    def _record_info(self, info: _FasterWhisperInfo) -> None:
        """Record detected language and duration metadata from ``info``."""
        language = getattr(info, "language", None)
        if isinstance(language, str) and language:
            self.backend_configuration["detected_language"] = language
            probability = getattr(info, "language_probability", None)
            if isinstance(probability, int | float):
                self.backend_metrics["detected_language_probability"] = float(probability)
        duration = getattr(info, "duration", None)
        if isinstance(duration, int | float):
            self.backend_metrics["audio_duration_seconds"] = float(duration)
        duration_after_vad = getattr(info, "duration_after_vad", None)
        if isinstance(duration_after_vad, int | float):
            self.backend_metrics["duration_after_vad_seconds"] = float(duration_after_vad)

    def _load(self) -> _FasterWhisperModel:
        if self._model is not None:
            return self._model
        try:
            from importlib.metadata import PackageNotFoundError, version

            def installed(package: str) -> str:
                try:
                    return version(package)
                except PackageNotFoundError:
                    return "unknown"

            model_path = resolve_model_path(self.model_reference)
            self.backend_models["model_path"] = model_path
            self.runtime_provenance = RuntimeProvenance(
                name="faster-whisper",
                version=installed("faster-whisper"),
                components={
                    "ctranslate2": installed("ctranslate2"),
                    "huggingface_hub": installed("huggingface-hub"),
                },
            )
            self._model = _create_whisper_model(model_path, self.device, self.compute_type)
            return self._model
        except Exception as error:
            raise ModelLoadError(
                f"could not load faster-whisper model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop the CTranslate2 model reference for sequential GPU execution."""
        self._model = None


def _create_whisper_model(model_path: str, device: str, compute_type: str) -> _FasterWhisperModel:
    """Construct the CTranslate2 model, importing the runtime lazily."""
    from faster_whisper import WhisperModel

    return cast(
        _FasterWhisperModel,
        WhisperModel(model_path, device=device, compute_type=compute_type),
    )


def whisper_language(language: str | None) -> str | None:
    """Reduce a locale like ``de-DE`` to the Whisper base code (``de``).

    ``None`` leaves language detection to the model.
    """
    return normalize_language(language)


def resolve_model_path(model: str) -> str:
    """Resolve a model reference to the active local Hugging Face snapshot.

    Hugging Face stores branch refs such as ``main`` under ``refs/`` and model
    contents under commit-named ``snapshots/`` directories. Resolve that ref
    explicitly so a cache with multiple historical snapshots remains
    deterministic in air-gapped operation. If no usable snapshot can be found
    while offline, fail instead of passing a repository-cache root to
    CTranslate2.
    """
    if Path(model).is_absolute():
        return model

    cache_root = os.environ.get("HF_HOME")
    if not cache_root:
        return model

    repo_dir = Path(cache_root) / "hub" / f"models--{model.replace('/', '--')}"
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    if not repo_dir.is_dir():
        if offline:
            raise ModelLoadError(
                f"required faster-whisper model {model!r} is not present in the offline cache "
                f"under {repo_dir}"
            )
        return model

    snapshots = repo_dir / "snapshots"
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        if revision and Path(revision).name == revision:
            snapshot = snapshots / revision
            if snapshot.is_dir():
                return str(snapshot)

    if snapshots.is_dir():
        revisions = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if len(revisions) == 1:
            return str(revisions[0])

    if offline:
        raise ModelLoadError(
            f"required faster-whisper model {model!r} has no resolvable cached snapshot "
            f"under {repo_dir}"
        )
    return model


def flatten_segment_words(segments: list[Any]) -> list[ASRWord]:
    """Map faster-whisper segment words into canonical ``ASRWord`` records.

    ``word.word`` may carry leading whitespace; it is stripped only to the
    extent required by the canonical transcript conventions, preserving
    punctuation and never concatenating words.
    """
    words: list[ASRWord] = []
    for segment in segments:
        segment_words = getattr(segment, "words", None)
        if segment_words is None:
            continue
        for word in segment_words:
            text = getattr(word, "word", None)
            if not isinstance(text, str) or not text.strip():
                continue
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if not isinstance(start, int | float) or not isinstance(end, int | float):
                raise ASROutputError("faster-whisper word is missing numeric start/end values")
            if end < start:
                raise ASROutputError("faster-whisper word ends before it starts")
            probability = getattr(word, "probability", None)
            confidence = float(probability) if isinstance(probability, int | float) else None
            words.append(
                ASRWord(
                    text=text.strip(),
                    start=float(start),
                    end=float(end),
                    confidence=confidence,
                )
            )
    return words
