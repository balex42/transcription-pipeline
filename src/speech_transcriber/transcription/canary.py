"""NVIDIA NeMo Canary 1B v2 adapter with native word timestamps.

The adapter restores the trusted ``.nemo`` checkpoint from an explicit local
path or a resolved Hugging Face cache snapshot. It deliberately never calls
``from_pretrained()`` during recognition, so air-gapped runs cannot fall back
to an online model lookup.

Recognition always uses one inference path: the normalized WAV is split into
deterministic, non-overlapping PCM chunks by exact frame arithmetic and each
chunk is transcribed sequentially with native timestamps rebased to
recording-global positions. Recordings shorter than one chunk produce a single
short chunk through the identical machinery.
"""

from __future__ import annotations

import logging
import tempfile
import time
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.language import normalize_language
from speech_transcriber.model_cache import resolve_hf_snapshot
from speech_transcriber.models import ASRWord, NormalizedAudio
from speech_transcriber.transcription import nemo_support
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities

LOGGER = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class CanaryChunk:
    """One exact-frame PCM chunk with its recording-global start offset."""

    path: Path
    start_frame: int
    frame_count: int
    sample_rate: int

    @property
    def offset_seconds(self) -> float:
        return self.start_frame / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


class CanaryTranscriber(Transcriber):
    """Transcribe normalized PCM chunks with NeMo Canary 1B v2."""

    capabilities = TranscriberCapabilities(True, True, True, True)

    def __init__(
        self,
        model: str,
        device: str,
        language: str,
        chunk_duration_seconds: float,
        working_directory: Path | None = None,
    ) -> None:
        if chunk_duration_seconds <= 0:
            raise ValueError("Canary chunk duration must be positive")
        self.model_reference = model
        self.device = device
        self.requested_language = language
        self.source_language = canary_language(language)
        self.target_language = self.source_language
        self.chunk_duration_seconds = chunk_duration_seconds
        self.working_directory = working_directory
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
            "inference_mode": "sequential_non_overlapping_chunks",
            "chunk_duration_seconds": chunk_duration_seconds,
        }
        self.runtime_provenance = nemo_support.initial_runtime_provenance()

    def load(self) -> None:
        """Restore the local NeMo checkpoint without contacting Hugging Face."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Transcribe every chunk sequentially and rebase timestamps globally."""
        model = self._load()
        words: list[ASRWord] = []
        chunk_count = 0
        transcription_seconds = 0.0
        try:
            with tempfile.TemporaryDirectory(
                prefix="canary-chunks.", dir=self._chunk_directory()
            ) as chunk_directory:
                for chunk, plan in create_canary_chunks(audio.path, self.chunk_duration_seconds):
                    chunk_path = write_canary_chunk(chunk_directory, chunk, audio.path)
                    chunk_count = plan["chunk_count"]
                    started = time.monotonic()
                    outputs = self._transcribe_chunk(model, chunk_path=chunk_path)
                    transcription_seconds += time.monotonic() - started
                    words.extend(
                        flatten_canary_words(outputs, offset_seconds=chunk.offset_seconds)
                    )
                    del outputs
        except (ASROutputError, ModelLoadError):
            raise
        except Exception as error:
            raise ASROutputError(
                "Canary recognition failed for "
                f"{audio.path} with model {self.model_reference}: {error}"
            ) from error
        self.backend_metrics["word_count"] = float(len(words))
        self.backend_metrics["chunk_count"] = float(chunk_count)
        self.backend_metrics["chunk_duration_seconds"] = self.chunk_duration_seconds
        self.backend_metrics["chunk_transcription_seconds"] = transcription_seconds
        self.backend_configuration["chunk_count"] = chunk_count
        validate_canary_words(words)
        return words

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
        self.runtime_provenance = nemo_support.nemo_runtime_provenance()
        return self._model

    def _transcribe_chunk(self, model: Any, *, chunk_path: Path) -> Any:
        try:
            return model.transcribe(
                [str(chunk_path)],
                batch_size=1,
                return_hypotheses=True,
                source_lang=self.source_language,
                target_lang=self.target_language,
                timestamps=True,
            )
        except Exception as error:
            raise ASROutputError(
                f"Canary recognition failed for chunk {chunk_path}: {error}"
            ) from error

    def _chunk_directory(self) -> Path:
        root = self.working_directory if self.working_directory is not None else Path.cwd() / "work"
        root.mkdir(parents=True, exist_ok=True)
        return root

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
    resolved from the active Hugging Face cache through the shared snapshot
    helper via ``refs/main`` so multiple historical snapshots remain
    deterministic. Resolution is always strict: runtime model downloading is
    intentionally unsupported, so a missing or ambiguous cache fails instead
    of returning a Hub repository ID; prefetch the repository before
    air-gapped use.
    """
    configured = Path(model).expanduser()
    if configured.is_absolute() or configured.exists():
        return _canary_model_file(configured)

    snapshot = resolve_hf_snapshot(model, subject="Canary model")
    return _canary_model_file(snapshot)


def flatten_canary_words(outputs: Sequence[Any], *, offset_seconds: float = 0.0) -> list[ASRWord]:
    """Map ``Hypothesis.timestamp['word']`` records to canonical ASR words.

    Local Canary timestamps are rebased by ``offset_seconds`` so every call
    yields recording-global timestamps.
    """
    if len(outputs) != 1:
        raise ASROutputError(f"Canary returned {len(outputs)} hypotheses for one recording")
    try:
        records = nemo_support.word_timestamp_records(outputs[0], subject="Canary")
    except ValueError as error:
        raise ASROutputError(str(error)) from error

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
        if start < 0 or end < start:
            raise ASROutputError("Canary word timestamp is outside the chunk")
        # NeMo's Canary timestamp records expose no stable per-word confidence.
        words.append(
            ASRWord(
                text=text.strip(),
                start=offset_seconds + float(start),
                end=offset_seconds + float(end),
                confidence=None,
            )
        )
    return words


def validate_canary_words(words: Sequence[ASRWord], *, tolerance: float = 1e-3) -> None:
    """Reject impossible global word timestamps or non-monotonic output."""
    previous_end: float | None = None
    for index, word in enumerate(words):
        start = word.start
        if start is None:
            raise ASROutputError(
                f"Canary global word timestamp is missing a start at index {index}: "
                f"'{word.text}'"
            )
        if start < -tolerance or word.end < start - tolerance:
            raise ASROutputError(
                f"Canary global word timestamp is invalid at index {index}: "
                f"'{word.text}' {start}-{word.end}"
            )
        if previous_end is not None and word.end < previous_end - tolerance:
            raise ASROutputError(
                "Canary global word timestamps reset or reorder across chunks at "
                f"index {index}: '{word.text}' {word.start}-{word.end}"
            )
        previous_end = word.end


def _canary_sample_rate() -> int:
    """Canary's checkpoint expects the pipeline's fixed 16 kHz normalized audio."""
    return 16000


def frames_per_chunk(chunk_duration_seconds: float, sample_rate: int) -> int:
    """Convert a chunk duration to an exact PCM frame count."""
    if chunk_duration_seconds <= 0:
        raise ValueError("chunk duration must be positive")
    return round(chunk_duration_seconds * sample_rate)


def create_canary_chunks(
    wav_path: Path, chunk_duration_seconds: float
) -> list[tuple[CanaryChunk, dict[str, int]]]:
    """Return the deterministic non-overlapping chunk plan for one WAV.

    Chunk boundaries use exact PCM frame arithmetic so chunk offsets never
    accumulate floating-point duration drift. A recording shorter than one
    chunk still yields exactly one short chunk.
    """
    with wave.open(str(wav_path), "rb") as source:
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        if (
            sample_rate != _canary_sample_rate()
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
        ):
            raise ASROutputError(
                "Canary chunking requires 16 kHz mono 16-bit PCM WAV; "
                f"{wav_path} does not match"
            )
    per_chunk = frames_per_chunk(chunk_duration_seconds, sample_rate)
    chunks = [
        CanaryChunk(
            path=wav_path,
            start_frame=start,
            frame_count=min(per_chunk, frame_count - start),
            sample_rate=sample_rate,
        )
        for start in range(0, frame_count, per_chunk)
    ]
    if not chunks:
        # A zero-frame WAV still flows through the single-chunk pipeline.
        chunks.append(
            CanaryChunk(path=wav_path, start_frame=0, frame_count=0, sample_rate=sample_rate)
        )
    plan = {
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "frames_per_chunk": per_chunk,
        "chunk_count": len(chunks),
    }
    return [(chunk, plan) for chunk in chunks]


def write_canary_chunk(directory: str | Path, chunk: CanaryChunk, source_path: Path) -> Path:
    """Write one chunk WAV by copying its exact PCM frame range from the source."""
    destination = Path(directory) / f"chunk-{chunk.start_frame:012d}.wav"
    with wave.open(str(source_path), "rb") as source:
        source.setpos(chunk.start_frame)
        frames = source.readframes(chunk.frame_count)
        params: tuple[Any, ...] = source.getparams()
    with wave.open(str(destination), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)
    return destination


def _canary_model_file(path: Path) -> str:
    return nemo_support.model_file(path, CANARY_MODEL_FILE, subject="Canary model")


def _restore_canary_model(model_path: str, device: str) -> Any:
    """Restore and place Canary on the requested device, importing NeMo lazily."""
    return nemo_support.restore_model(model_path, device, subject="Canary")