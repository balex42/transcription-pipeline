"""Native Transformers adapter for NVIDIA Parakeet TDT."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, cast

from speech_transcriber.audio.segmenter import AudioSegmenter
from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.models import ASRWord, AudioSegment, NormalizedAudio
from speech_transcriber.runtime.device import inference_dtype
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities
from speech_transcriber.transcription.segments import reconcile_segment_words


class _ParakeetProcessor(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> Any: ...

    def decode(self, token_ids: object, **kwargs: object) -> tuple[str, object]: ...


class _ParakeetModel(Protocol):
    def to(self, device: str) -> object: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> Any: ...


class ParakeetTranscriber(Transcriber):
    """Generate punctuation-preserving words and native TDT timestamps."""

    capabilities = TranscriberCapabilities(True, True, True, True)

    def __init__(
        self,
        model: str,
        device: str,
        segment_duration: float = 180.0,
        segment_overlap: float = 15.0,
    ) -> None:
        self.model_reference = model
        self.device = device
        _, self.dtype_name = inference_dtype(device)
        self._model: _ParakeetModel | None = None
        self._processor: _ParakeetProcessor | None = None
        self._segmenter = AudioSegmenter(segment_duration, segment_overlap)
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        self.backend_configuration = {
            "segment_duration_seconds": segment_duration,
            "segment_overlap_seconds": segment_overlap,
        }

    def load(self) -> None:
        """Load the native Parakeet TDT model lazily."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Segment and reconcile the full recording inside the Parakeet adapter."""
        segments = self._segmenter.segment(audio)
        self.backend_metrics["segments_processed"] = float(len(segments))
        return reconcile_segment_words(
            segments, {segment.index: self._transcribe_segment(segment) for segment in segments}
        )

    def _transcribe_segment(self, segment: AudioSegment) -> list[ASRWord]:
        """Return native start/end timestamps relative to one internal segment."""
        model, processor = self._load()
        import torch

        dtype, _ = inference_dtype(self.device)
        inputs = processor(segment.audio, sampling_rate=segment.sample_rate, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True, do_sample=False)
        _, timestamps = processor.decode(
            output.sequences,
            durations=output.durations,
            skip_special_tokens=True,
        )
        return normalize_parakeet_timestamps(timestamps)

    def _load(self) -> tuple[_ParakeetModel, _ParakeetProcessor]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import AutoModelForTDT, AutoProcessor

            dtype, _ = inference_dtype(self.device)
            processor = cast(
                _ParakeetProcessor,
                AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                    self.model_reference, trust_remote_code=False
                ),
            )
            model = cast(
                _ParakeetModel,
                AutoModelForTDT.from_pretrained(
                    self.model_reference, dtype=dtype, trust_remote_code=False
                ),
            )
            model.to(self.device)
            model.eval()
            self._model, self._processor = model, processor
            return model, processor
        except Exception as error:
            raise ModelLoadError(
                f"could not load Parakeet model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop model references for sequential GPU execution."""
        self._model = None
        self._processor = None


def normalize_parakeet_timestamps(timestamps: object) -> list[ASRWord]:
    """Normalize Parakeet pieces into punctuation-preserving word intervals."""
    entries = list(_timestamp_entries(timestamps))
    if entries and all("token" in entry and "word" not in entry for entry in entries):
        return _token_pieces_to_words(entries)
    words: list[ASRWord] = []
    for entry in entries:
        text = str(entry.get("word", entry.get("token", entry.get("text", "")))).strip()
        bounds = entry.get("timestamp")
        start = entry.get("start")
        end = entry.get("end")
        if isinstance(bounds, tuple | list) and len(bounds) == 2:
            start, end = bounds
        if not text:
            continue
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ASROutputError("Parakeet word timestamp is missing numeric start/end values")
        if end < start:
            raise ASROutputError("Parakeet word timestamp ends before it starts")
        words.append(ASRWord(text=text, start=float(start), end=float(end)))
    return words


def _token_pieces_to_words(entries: list[dict[str, object]]) -> list[ASRWord]:
    """Join SentencePiece-style timestamped fragments into lexical words."""
    words: list[ASRWord] = []
    text = ""
    start: float | None = None
    end: float | None = None
    punctuation = {"?", "'", "!", "-", ":", ",", "%", "/", "."}

    def emit() -> None:
        nonlocal text, start, end
        if text and start is not None and end is not None:
            words.append(ASRWord(text=text, start=start, end=end))
        text, start, end = "", None, None

    for entry in entries:
        piece = str(entry["token"])
        raw_start, raw_end = entry.get("start"), entry.get("end")
        if not isinstance(raw_start, int | float) or not isinstance(raw_end, int | float):
            raise ASROutputError("Parakeet token timestamp is missing numeric start/end values")
        if raw_end < raw_start:
            raise ASROutputError("Parakeet token timestamp ends before it starts")
        stripped = piece.strip()
        if not stripped:
            continue
        begins_word = bool(piece[:1].isspace())
        if begins_word and text:
            emit()
        if stripped in punctuation and text:
            text += stripped
            end = max(end or float(raw_end), float(raw_end))
            continue
        if not text:
            start = float(raw_start)
        text += stripped
        end = float(raw_end)
    emit()
    return words


def _timestamp_entries(timestamps: object) -> Iterable[dict[str, object]]:
    if isinstance(timestamps, dict):
        for key in ("words", "word", "timestamps"):
            nested = timestamps.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [timestamps]
    if isinstance(timestamps, list):
        if len(timestamps) == 1 and isinstance(timestamps[0], list):
            return [item for item in timestamps[0] if isinstance(item, dict)]
        return [item for item in timestamps if isinstance(item, dict)]
    raise ASROutputError("Parakeet returned an unsupported timestamp structure")
