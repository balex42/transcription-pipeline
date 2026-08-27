"""Native Hugging Face Transformers adapter for Granite Speech Plus."""

from __future__ import annotations

import logging
from typing import Any, Protocol, cast

from meeting_transcriber.errors import ModelLoadError
from meeting_transcriber.models import ASRWord, AudioChunk
from meeting_transcriber.runtime.device import inference_dtype
from meeting_transcriber.transcription.base import Transcriber, TranscriberCapabilities
from meeting_transcriber.transcription.timestamp_parser import parse_timestamped_words

LOGGER = logging.getLogger(__name__)


class _GraniteProcessor(Protocol):
    def apply_chat_template(self, conversation: object, **kwargs: object) -> str: ...

    def __call__(self, *args: object, **kwargs: object) -> Any: ...

    def decode(self, token_ids: object, **kwargs: object) -> str: ...


class _GraniteModel(Protocol):
    def to(self, device: str) -> object: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> Any: ...


class GraniteTranscriber(Transcriber):
    """Generate deterministic German timestamp-mode transcription with Granite."""

    capabilities = TranscriberCapabilities(False, True, False, False)

    def __init__(
        self,
        model: str,
        device: str,
        system_prompt: str,
        timestamp_prompt: str,
        max_new_tokens: int = 10_000,
    ) -> None:
        self.model_reference = model
        self.device = device
        self.system_prompt = system_prompt
        self.timestamp_prompt = timestamp_prompt
        self.max_new_tokens = max_new_tokens
        _, self.dtype_name = inference_dtype(device)
        self._model: _GraniteModel | None = None
        self._processor: _GraniteProcessor | None = None

    def transcribe(self, chunk: AudioChunk) -> list[ASRWord]:
        """Transcribe one chunk and parse its chunk-relative Granite timestamps."""
        model, processor = self._load()
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.timestamp_prompt},
        ]
        prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            prompt,
            chunk.audio,
            sampling_rate=chunk.sample_rate,
            device=self.device,
            return_tensors="pt",
        ).to(self.device)
        import torch

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        decoded = processor.decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
        return parse_timestamped_words(decoded, chunk.chunk_id)

    def load(self) -> None:
        """Load Granite so callers can measure initialization separately."""
        self._load()

    def _load(self) -> tuple[_GraniteModel, _GraniteProcessor]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            dtype, _ = inference_dtype(self.device)
            LOGGER.info(
                "loading Granite model",
                extra={"model": self.model_reference, "device": self.device},
            )
            processor = cast(
                _GraniteProcessor,
                AutoProcessor.from_pretrained(self.model_reference),  # type: ignore[no-untyped-call]
            )
            model = cast(
                _GraniteModel,
                AutoModelForSpeechSeq2Seq.from_pretrained(self.model_reference, dtype=dtype),
            )
            model.to(self.device)
            model.eval()
            self._model, self._processor = model, processor
            LOGGER.info("loaded Granite model")
            return model, processor
        except Exception as error:
            raise ModelLoadError(
                f"could not load Granite model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop model and processor references for lifecycle cleanup."""
        self._model = None
        self._processor = None
