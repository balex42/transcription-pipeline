"""Native Transformers recognition stage for Cohere Transcribe."""

from __future__ import annotations

from typing import Any

from speech_transcriber.errors import CohereRecognitionError, ModelLoadError
from speech_transcriber.models import AudioSegment
from speech_transcriber.runtime.device import inference_dtype


class CohereRecognizer:
    """Recognize bounded audio before a separate timing pass."""

    def __init__(
        self,
        model: str,
        device: str,
        language: str = "de-DE",
        punctuation: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_reference = model
        self.device = device
        self.language = language
        self.punctuation = punctuation
        self.max_new_tokens = max_new_tokens
        _, self.dtype_name = inference_dtype(device)
        self._model: Any | None = None
        self._processor: Any | None = None

    def load(self) -> None:
        """Load Cohere's native Transformers implementation without remote code."""
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration

            dtype, _ = inference_dtype(self.device)
            processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_reference, trust_remote_code=False
            )
            model = CohereAsrForConditionalGeneration.from_pretrained(
                self.model_reference,
                dtype=dtype,
                trust_remote_code=False,
            )
            model.to(self.device)  # type: ignore[arg-type]
            model.eval()  # type: ignore[no-untyped-call]
            self._model, self._processor = model, processor
        except Exception as error:
            raise ModelLoadError(
                f"could not load Cohere ASR model {self.model_reference}: {error}"
            ) from error

    def recognize(self, segment: AudioSegment) -> str:
        """Return deterministic transcription text for one internal segment."""
        self.load()
        assert self._model is not None and self._processor is not None
        try:
            import torch

            dtype, _ = inference_dtype(self.device)
            language = _cohere_language(self.language)
            inputs = self._processor(
                segment.audio,
                sampling_rate=segment.sample_rate,
                language=language,
                punctuation=self.punctuation,
                return_tensors="pt",
            )
            audio_chunk_index = inputs.get("audio_chunk_index")
            inputs = inputs.to(self.device, dtype=dtype)
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                )
            decoded = self._processor.decode(
                output_ids,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=language,
            )
            return _single_transcript(decoded)
        except CohereRecognitionError as error:
            raise CohereRecognitionError(
                "Cohere recognition failed for "
                f"segment {segment.index} ({segment.start:.3f}-{segment.end:.3f}s) "
                f"with model {self.model_reference}: {error}"
            ) from error
        except Exception as error:
            raise CohereRecognitionError(
                "Cohere recognition failed for "
                f"segment {segment.index} ({segment.start:.3f}-{segment.end:.3f}s) "
                f"with model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop Cohere model references before forced alignment."""
        self._model = None
        self._processor = None


def _cohere_language(language: str) -> str:
    """Reduce a locale like ``de-DE`` to Cohere's base code (``de``)."""
    return language.split("-", 1)[0].lower()


def _single_transcript(decoded: object) -> str:
    """Normalize the processor's one-segment decode result."""
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, list) and len(decoded) == 1 and isinstance(decoded[0], str):
        return decoded[0].strip()
    raise CohereRecognitionError("Cohere ASR returned an unsupported decoded response")
