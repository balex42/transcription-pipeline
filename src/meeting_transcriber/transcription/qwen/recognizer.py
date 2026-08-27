"""Native Transformers recognition stage for Qwen3-ASR."""

from __future__ import annotations

from typing import Any

from meeting_transcriber.errors import ModelLoadError, QwenRecognitionError
from meeting_transcriber.models import AudioSegment
from meeting_transcriber.runtime.device import inference_dtype


class QwenRecognizer:
    """Recognize German chunk text before a separate timing pass."""

    def __init__(
        self,
        model: str,
        device: str,
        context: str | None = None,
        max_new_tokens: int = 2_048,
    ) -> None:
        self.model_reference = model
        self.device = device
        self.context = context
        self.max_new_tokens = max_new_tokens
        _, self.dtype_name = inference_dtype(device)
        self._model: Any | None = None
        self._processor: Any | None = None

    def load(self) -> None:
        """Load the native Qwen3-ASR implementation without remote code."""
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            dtype, _ = inference_dtype(self.device)
            processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_reference, trust_remote_code=False
            )
            model = AutoModelForMultimodalLM.from_pretrained(
                self.model_reference,
                dtype=dtype,
                trust_remote_code=False,
            )
            model.to(self.device)
            model.eval()
            self._model, self._processor = model, processor
        except Exception as error:
            raise ModelLoadError(
                f"could not load Qwen ASR model {self.model_reference}: {error}"
            ) from error

    def recognize(self, segment: AudioSegment) -> str:
        """Return deterministic German transcription text for one internal segment."""
        self.load()
        assert self._model is not None and self._processor is not None
        try:
            import torch

            dtype, _ = inference_dtype(self.device)
            request: dict[str, object] = {
                "audio": segment.audio,
                "language": "de",
            }
            if self.context:
                request["prompt"] = self.context
            inputs = self._processor.apply_transcription_request(**request)
            inputs = inputs.to(self.device, dtype=dtype)
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            parsed = self._processor.decode(generated_ids, return_format="parsed")
            if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
                raise QwenRecognitionError("Qwen ASR returned an unsupported decoded response")
            text = parsed[0].get("transcription")
            if not isinstance(text, str):
                raise QwenRecognitionError("Qwen ASR response did not contain transcription text")
            return text.strip()
        except QwenRecognitionError as error:
            raise QwenRecognitionError(
                "Qwen recognition failed for "
                f"segment {segment.index} ({segment.start:.3f}-{segment.end:.3f}s) "
                f"with model {self.model_reference}: {error}"
            ) from error
        except Exception as error:
            raise QwenRecognitionError(
                "Qwen recognition failed for "
                f"segment {segment.index} ({segment.start:.3f}-{segment.end:.3f}s) "
                f"with model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop Qwen ASR model references before forced alignment."""
        self._model = None
        self._processor = None
