"""Cache-aware native Transformers adapter for Mistral Voxtral Realtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from speech_transcriber.audio.segmenter import load_normalized_samples
from speech_transcriber.errors import ModelLoadError, VoxtralStreamingError, VoxtralTimestampError
from speech_transcriber.models import ASRWord, NormalizedAudio
from speech_transcriber.runtime.device import inference_dtype
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities
from speech_transcriber.transcription.voxtral.timestamps import parse_voxtral_words


class _VoxtralTokenizer(Protocol):
    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = False
    ) -> list[str]: ...


class _VoxtralProcessor(Protocol):
    num_right_pad_tokens: int
    raw_audio_length_per_tok: int
    num_samples_first_audio_chunk: int
    num_samples_per_audio_chunk: int
    num_mel_frames_first_audio_chunk: int
    audio_length_per_tok: int
    num_delay_tokens: int
    feature_extractor: object
    tokenizer: _VoxtralTokenizer

    def __call__(self, audio: object, **kwargs: object) -> Any: ...

    def decode(self, token_ids: list[int], **kwargs: object) -> object: ...


class _VoxtralModel(Protocol):
    def to(self, device: str) -> object: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> Any: ...


@dataclass
class VoxtralStreamingState:
    """Per-recording transport bookkeeping; native generate owns all model caches."""

    buffers_processed: int = 0
    samples_submitted: int = 0


class VoxtralTranscriber(Transcriber):
    """Use Voxtral's single native streaming generation session for a whole recording."""

    capabilities = TranscriberCapabilities(False, True, True, True, streaming=True)

    def __init__(self, model: str, device: str) -> None:
        self.model_reference = model
        self.device = device
        _, self.dtype_name = inference_dtype(device)
        self._model: _VoxtralModel | None = None
        self._processor: _VoxtralProcessor | None = None
        self._state = VoxtralStreamingState()
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        self.backend_configuration: dict[str, str | int | float | bool | None] = {
            "streaming": True,
            "timestamps": "streaming_word_end_proxy",
            "temperature": 0.0,
        }

    def load(self) -> None:
        """Load the native Transformers model and its official processor."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Submit processor-sized buffers to one cache-aware native generation call."""
        model, processor = self._load()
        self._state = VoxtralStreamingState()
        try:
            import torch

            samples = load_normalized_samples(audio)
            padded = np.pad(
                samples,
                (
                    0,
                    _processor_int(processor, "num_right_pad_tokens")
                    * _processor_int(processor, "raw_audio_length_per_tok"),
                ),
            )
            first_size = _processor_int(processor, "num_samples_first_audio_chunk")
            initial, first_features = self._prepare_buffer(
                processor,
                _padded(padded[:first_size], first_size),
                audio.metadata.sample_rate,
                True,
            )
            initial.pop("input_features", None)
            prompt_ids = _tensor_tokens(initial.get("input_ids"))
            if not prompt_ids:
                raise VoxtralStreamingError("Voxtral processor did not return initial input IDs")
            with torch.inference_mode():
                output = model.generate(
                    **initial,
                    input_features=self._stream_features(
                        processor, padded, first_size, first_features, audio.metadata.sample_rate
                    ),
                    # Transformers represents temperature zero as greedy decoding.
                    do_sample=False,
                    return_dict_in_generate=True,
                )
            generated = _generated_tokens(output, prompt_ids)
            pieces = processor.tokenizer.convert_ids_to_tokens(generated)
            if not all(isinstance(piece, str) for piece in pieces):
                raise VoxtralTimestampError("Voxtral tokenizer returned non-text token pieces")
            seconds_per_token = (
                _processor_int(processor, "raw_audio_length_per_tok") / audio.metadata.sample_rate
            )
            for name in (
                "native_emission_groups",
                "multi_word_emission_groups",
                "inferred_final_emission_groups",
                "inferred_final_words",
            ):
                self.backend_metrics[name] = 0.0
            words = parse_voxtral_words(
                generated,
                pieces,
                lambda ids: processor.decode(ids, skip_special_tokens=True),
                _processor_int(processor, "num_delay_tokens", allow_zero=True),
                seconds_per_token,
                audio.metadata.duration_seconds,
                self.backend_metrics,
            )
            self.backend_metrics["stream_buffers_processed"] = float(self._state.buffers_processed)
            self.backend_metrics["stream_samples_submitted"] = float(self._state.samples_submitted)
            self.backend_configuration["num_delay_tokens"] = _processor_int(
                processor, "num_delay_tokens", allow_zero=True
            )
            self.backend_configuration["num_right_pad_tokens"] = _processor_int(
                processor, "num_right_pad_tokens", allow_zero=True
            )
            return words
        except (VoxtralStreamingError, VoxtralTimestampError):
            raise
        except Exception as error:
            raise VoxtralStreamingError(
                "Voxtral streaming failed for "
                f"{audio.path} with model {self.model_reference}: {error}"
            ) from error

    def _prepare_buffer(
        self,
        processor: _VoxtralProcessor,
        samples: NDArray[np.float32],
        sample_rate: int,
        first: bool,
    ) -> tuple[dict[str, object], object]:
        """Feature-extract one processor-defined streaming transport buffer."""
        dtype, _ = inference_dtype(self.device)
        inputs = processor(
            samples,
            sampling_rate=sample_rate,
            is_streaming=True,
            is_first_audio_chunk=first,
            return_tensors="pt",
        )
        moved = inputs.to(self.device, dtype=dtype)
        features = moved.get("input_features")
        if features is None:
            raise VoxtralStreamingError("Voxtral processor did not return input_features")
        return cast(dict[str, object], moved), features

    def _stream_features(
        self,
        processor: _VoxtralProcessor,
        samples: NDArray[np.float32],
        first_size: int,
        first_features: object,
        sample_rate: int,
    ) -> object:
        """Yield continuous feature buffers while native generate carries all caches."""
        self._state.buffers_processed += 1
        self._state.samples_submitted += first_size
        yield first_features
        mel_frame_index = _processor_int(processor, "num_mel_frames_first_audio_chunk")
        hop_length = _attribute_int(processor.feature_extractor, "hop_length")
        win_length = _attribute_int(processor.feature_extractor, "win_length")
        buffer_size = _processor_int(processor, "num_samples_per_audio_chunk")
        while True:
            start = mel_frame_index * hop_length - win_length // 2
            if start >= len(samples):
                return
            buffer = _padded(samples[start : start + buffer_size], buffer_size)
            _, features = self._prepare_buffer(processor, buffer, sample_rate, False)
            self._state.buffers_processed += 1
            self._state.samples_submitted += buffer_size
            yield features
            mel_frame_index += _processor_int(processor, "audio_length_per_tok")

    def _load(self) -> tuple[_VoxtralModel, _VoxtralProcessor]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

            dtype, _ = inference_dtype(self.device)
            processor = cast(
                _VoxtralProcessor,
                AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                    self.model_reference, trust_remote_code=False
                ),
            )
            model = cast(
                _VoxtralModel,
                VoxtralRealtimeForConditionalGeneration.from_pretrained(
                    self.model_reference, dtype=dtype, trust_remote_code=False
                ),
            )
            model.to(self.device)
            model.eval()
            self._model, self._processor = model, processor
            return model, processor
        except Exception as error:
            raise ModelLoadError(
                f"could not load Voxtral model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop model state so one GPU can serve the next backend safely."""
        self._model = None
        self._processor = None
        self._state = VoxtralStreamingState()


def _generated_tokens(output: object, prompt_ids: list[int]) -> list[int]:
    sequences = getattr(output, "sequences", None)
    sequence = _tensor_tokens(sequences)
    if sequence[: len(prompt_ids)] != prompt_ids:
        raise VoxtralStreamingError(
            "Voxtral generation did not preserve the processor prompt token prefix"
        )
    return sequence[len(prompt_ids) :]


def _tensor_tokens(value: object) -> list[int]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list) or not all(isinstance(token, int) for token in raw):
        return []
    return raw


def _processor_int(processor: _VoxtralProcessor, name: str, allow_zero: bool = False) -> int:
    return _attribute_int(processor, name, allow_zero)


def _attribute_int(source: object, name: str, allow_zero: bool = False) -> int:
    value = getattr(source, name, None)
    if callable(value):
        value = value()
    if not isinstance(value, int) or value < 0 or (value == 0 and not allow_zero):
        raise VoxtralStreamingError(f"Voxtral processor did not expose a valid {name}")
    return value


def _padded(samples: NDArray[np.float32], size: int) -> NDArray[np.float32]:
    if len(samples) >= size:
        return samples
    return np.pad(samples, (0, size - len(samples)))
