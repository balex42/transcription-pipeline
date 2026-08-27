"""Cache-aware native Transformers adapter for NVIDIA Nemotron 3.5 ASR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from meeting_transcriber.audio.segmenter import load_normalized_samples
from meeting_transcriber.errors import ASROutputError, ModelLoadError, NemotronStreamingError
from meeting_transcriber.models import ASRWord, NormalizedAudio
from meeting_transcriber.runtime.device import inference_dtype
from meeting_transcriber.transcription.base import Transcriber, TranscriberCapabilities


class _NemotronProcessor(Protocol):
    default_num_lookahead_tokens: int
    num_samples_first_audio_chunk: int
    num_samples_per_audio_chunk: int
    num_mel_frames_first_audio_chunk: int
    num_mel_frames_per_audio_chunk: int
    streaming_latency_ms: int
    feature_extractor: object

    def __call__(self, audio: object, **kwargs: object) -> Any: ...

    def decode(self, sequences: object, **kwargs: object) -> object: ...

    def set_num_lookahead_tokens(self, value: int) -> None: ...


class _NemotronModel(Protocol):
    def to(self, device: str) -> object: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> Any: ...


@dataclass
class NemotronStreamingState:
    """Per-meeting streaming bookkeeping; native generate owns model caches."""

    buffers_processed: int = 0
    samples_submitted: int = 0


class NemotronTranscriber(Transcriber):
    """Use native RNNT generator streaming for one continuous meeting timeline."""

    capabilities = TranscriberCapabilities(True, True, True, True, streaming=True)

    def __init__(
        self, model: str, device: str, language: str = "de-DE", lookahead: int | None = None
    ) -> None:
        self.model_reference = model
        self.device = device
        self.language = language
        self.requested_lookahead = lookahead
        _, self.dtype_name = inference_dtype(device)
        self._model: _NemotronModel | None = None
        self._processor: _NemotronProcessor | None = None
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        self.backend_configuration: dict[str, str | int | float | bool | None] = {
            "language": language,
            "num_lookahead_tokens": lookahead,
            "streaming": True,
        }
        self._state = NemotronStreamingState()

    def load(self) -> None:
        """Load the native RNNT model without NeMo or remote code."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Feed processor-derived streaming buffers into cache-aware native generation."""
        model, processor = self._load()
        self._state = NemotronStreamingState()
        try:
            import torch

            lookahead = self._configure_lookahead(processor)
            samples = load_normalized_samples(audio)
            first_size = _processor_int(processor, "num_samples_first_audio_chunk")
            initial, first_features = self._prepare_buffer(
                processor,
                _padded(samples[:first_size], first_size),
                audio.metadata.sample_rate,
                True,
            )
            initial.pop("input_features", None)
            with torch.inference_mode():
                output = model.generate(
                    **initial,
                    input_features=self._stream_features(
                        processor, samples, first_size, first_features, audio.metadata.sample_rate
                    ),
                    return_dict_in_generate=True,
                )
            decoded = processor.decode(
                output.sequences, durations=output.durations, skip_special_tokens=True
            )
            token_timestamps = _token_timestamps(decoded)
            words = aggregate_nemotron_tokens(token_timestamps, audio.metadata.duration_seconds)
            self.backend_metrics["stream_buffers_processed"] = float(self._state.buffers_processed)
            self.backend_metrics["stream_samples_submitted"] = float(self._state.samples_submitted)
            self.backend_configuration["num_lookahead_tokens"] = lookahead
            self.backend_configuration["streaming_latency_ms"] = getattr(
                processor, "streaming_latency_ms", None
            )
            return words
        except NemotronStreamingError:
            raise
        except Exception as error:
            raise NemotronStreamingError(
                "Nemotron streaming failed for "
                f"{audio.path} with model {self.model_reference}: {error}"
            ) from error

    def _prepare_buffer(
        self,
        processor: _NemotronProcessor,
        samples: NDArray[np.float32],
        sample_rate: int,
        first: bool,
    ) -> tuple[dict[str, object], object]:
        """Feature-extract one model-sized transport buffer."""
        dtype, _ = inference_dtype(self.device)
        inputs = processor(
            samples,
            sampling_rate=sample_rate,
            language=self.language,
            is_streaming=True,
            is_first_audio_chunk=first,
            return_tensors="pt",
        )
        moved = inputs.to(self.device, dtype=dtype)
        features = moved.get("input_features")
        if features is None:
            raise NemotronStreamingError("Nemotron processor did not return input_features")
        return cast(dict[str, object], moved), features

    def _stream_features(
        self,
        processor: _NemotronProcessor,
        samples: NDArray[np.float32],
        first_size: int,
        first_features: object,
        sample_rate: int,
    ) -> object:
        """Yield model-sized feature buffers; native generate carries all ASR caches."""
        first_frames = _processor_int(processor, "num_mel_frames_first_audio_chunk")
        frames_per_buffer = _processor_int(processor, "num_mel_frames_per_audio_chunk")
        buffer_size = _processor_int(processor, "num_samples_per_audio_chunk")
        feature_extractor = getattr(processor, "feature_extractor", None)
        hop_length = _attribute_int(feature_extractor, "hop_length")
        n_fft = _attribute_int(feature_extractor, "n_fft")
        self._state.buffers_processed += 1
        self._state.samples_submitted += first_size
        yield _first_streaming_features(first_features, first_frames)
        mel_frame_index = first_frames
        start = mel_frame_index * hop_length - n_fft // 2
        while start < len(samples):
            buffer = _padded(samples[start : start + buffer_size], buffer_size)
            _, features = self._prepare_buffer(processor, buffer, sample_rate, False)
            self._state.buffers_processed += 1
            self._state.samples_submitted += buffer_size
            yield features
            mel_frame_index += frames_per_buffer
            start = mel_frame_index * hop_length - n_fft // 2

    def _configure_lookahead(self, processor: _NemotronProcessor) -> int:
        """Use only processor-advertised lookahead values."""
        if self.requested_lookahead is not None:
            try:
                processor.set_num_lookahead_tokens(self.requested_lookahead)
            except Exception as error:
                raise NemotronStreamingError(
                    "Nemotron does not support "
                    f"{self.requested_lookahead} lookahead tokens: {error}"
                ) from error
        return _processor_int(processor, "default_num_lookahead_tokens", allow_zero=True)

    def _load(self) -> tuple[_NemotronModel, _NemotronProcessor]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import AutoModelForRNNT, AutoProcessor

            dtype, _ = inference_dtype(self.device)
            processor = cast(
                _NemotronProcessor,
                AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                    self.model_reference, trust_remote_code=False
                ),
            )
            model = cast(
                _NemotronModel,
                AutoModelForRNNT.from_pretrained(
                    self.model_reference, dtype=dtype, trust_remote_code=False
                ),
            )
            model.to(self.device)
            model.eval()
            self._model, self._processor = model, processor
            return model, processor
        except Exception as error:
            raise ModelLoadError(
                f"could not load Nemotron model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop model state so one GPU can serve the next backend safely."""
        self._model = None
        self._processor = None
        self._state = NemotronStreamingState()


def aggregate_nemotron_tokens(entries: list[dict[str, object]], duration: float) -> list[ASRWord]:
    """Aggregate native token emission intervals into lexical words.

    Leading whitespace or SentencePiece/BPE word markers start words. Trailing
    punctuation joins the preceding lexical word; opening punctuation joins the
    next one. Native RNNT emission times are approximate alignment boundaries.
    """
    words: list[ASRWord] = []
    text = ""
    start: float | None = None
    end: float | None = None
    opening = "([{'\"„«"
    trailing = ".,!?;:%)]}'\"”»"

    def emit() -> None:
        nonlocal text, start, end
        if text and start is not None and end is not None and start < duration:
            words.append(ASRWord(text=text, start=start, end=max(min(end, duration), start)))
        text, start, end = "", None, None

    for entry in entries:
        token = _timestamp_value(entry, "token")
        raw_start = _timestamp_value(entry, "start")
        raw_end = _timestamp_value(entry, "end")
        if not isinstance(token, str) or not isinstance(raw_start, int | float) or not isinstance(
            raw_end, int | float
        ):
            raise ASROutputError("Nemotron token timestamp is missing text or numeric bounds")
        start_time, end_time = float(raw_start), float(raw_end)
        if start_time < 0 or end_time < start_time:
            raise ASROutputError("Nemotron token timestamp is invalid")
        if token.startswith("<") and token.endswith(">"):
            continue
        starts_word = token[:1].isspace() or token.startswith(("▁", "Ġ"))
        piece = token.lstrip(" ▁Ġ")
        if not piece:
            continue
        if all(character in trailing for character in piece):
            if text:
                text += piece
                end = end_time
            continue
        if starts_word and text:
            emit()
        if not text:
            start = start_time
        if all(character in opening for character in piece):
            text += piece
        else:
            text += piece
        end = end_time
    emit()
    return _validate_global_words(words, duration)


def _token_timestamps(decoded: object) -> list[dict[str, object]]:
    if not isinstance(decoded, tuple) or len(decoded) != 2:
        raise NemotronStreamingError("Nemotron processor did not return token timestamps")
    timestamps = decoded[1]
    if isinstance(timestamps, list) and len(timestamps) == 1 and isinstance(timestamps[0], list):
        timestamps = timestamps[0]
    if not isinstance(timestamps, list) or not all(isinstance(item, dict) for item in timestamps):
        raise NemotronStreamingError("Nemotron returned an unsupported token timestamp structure")
    return cast(list[dict[str, object]], timestamps)


def _timestamp_value(entry: dict[str, object], name: str) -> object:
    return entry.get(name, entry.get(f"{name}_time"))


def _validate_global_words(words: list[ASRWord], duration: float) -> list[ASRWord]:
    previous_start = 0.0
    for word in words:
        if (
            word.start is None
            or word.end < word.start
            or word.start + 1e-6 < previous_start
            or word.end > duration + 0.25
        ):
            raise NemotronStreamingError("Nemotron returned non-monotonic global token timing")
        # RNNT can emit several non-blank tokens in one encoder frame. Their
        # resulting lexical intervals may overlap while their emission order is valid.
        previous_start = word.start
    return words


def _processor_int(processor: _NemotronProcessor, name: str, allow_zero: bool = False) -> int:
    return _attribute_int(processor, name, allow_zero)


def _attribute_int(source: object, name: str, allow_zero: bool = False) -> int:
    value = getattr(source, name, None)
    if not isinstance(value, int) or value < 0 or (value == 0 and not allow_zero):
        raise NemotronStreamingError(f"Nemotron processor did not expose a valid {name}")
    return value


def _first_streaming_features(features: object, frames: int) -> object:
    """Trim first processor output to the model-advertised first stream shape."""
    try:
        return cast(Any, features)[:, :frames, :]
    except (IndexError, TypeError):
        return features


def _padded(samples: NDArray[np.float32], size: int) -> NDArray[np.float32]:
    if len(samples) >= size:
        return samples
    return np.pad(samples, (0, size - len(samples)))
