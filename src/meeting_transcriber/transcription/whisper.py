"""Whisper recognition configured for reusable forced word alignment."""

from __future__ import annotations

from typing import Any, Protocol, cast

from meeting_transcriber.errors import ASROutputError, ModelLoadError
from meeting_transcriber.models import AudioSegment
from meeting_transcriber.runtime.device import inference_dtype
from meeting_transcriber.transcription.base import TranscriberCapabilities
from meeting_transcriber.transcription.forced_alignment import (
    ForcedAligner,
    ForcedAlignmentTranscriber,
)


class _WhisperModel(Protocol):
    def to(self, device: str) -> object: ...

    def eval(self) -> object: ...


class _WhisperPipeline(Protocol):
    def __call__(self, inputs: object, **kwargs: object) -> dict[str, object]: ...


class WhisperRecognizer:
    """Use Transformers Whisper for bounded German transcript recognition."""

    def __init__(self, model: str, device: str) -> None:
        self.model_reference = model
        self.device = device
        _, self.dtype_name = inference_dtype(device)
        self._model: _WhisperModel | None = None
        self._pipeline: _WhisperPipeline | None = None

    def load(self) -> None:
        """Load Whisper and its ASR pipeline lazily."""
        self._load()

    def recognize(self, segment: AudioSegment) -> str:
        """Return transcript text without retaining word-timestamp attention tensors."""
        pipeline = self._load()
        import torch

        with torch.inference_mode():
            result = pipeline(
                segment.audio,
                generate_kwargs={"language": "german", "task": "transcribe"},
            )
        return normalize_whisper_text(result)

    def _load(self) -> _WhisperPipeline:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            dtype, _ = inference_dtype(self.device)
            loaded_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_reference,
                dtype=dtype,
                trust_remote_code=False,
            )
            loaded_model.to(self.device)
            loaded_model.eval()
            processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_reference, trust_remote_code=False
            )
            device_index = 0 if self.device == "cuda" else -1
            asr_pipeline = cast(
                _WhisperPipeline,
                pipeline(
                    "automatic-speech-recognition",
                    model=loaded_model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    dtype=cast(Any, dtype),
                    device=device_index,
                ),
            )
            self._model, self._pipeline = cast(_WhisperModel, loaded_model), asr_pipeline
            return asr_pipeline
        except Exception as error:
            raise ModelLoadError(
                f"could not load Whisper model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop model and pipeline references before forced alignment."""
        self._model = None
        self._pipeline = None


class WhisperTranscriber(ForcedAlignmentTranscriber):
    """Recognize with Whisper and delegate word timing to any forced aligner."""

    capabilities = TranscriberCapabilities(True, True, True, True, requires_forced_alignment=True)

    def __init__(self, model: str, device: str, aligner: ForcedAligner) -> None:
        # Whisper accepts at most 30 seconds per feature window. Recognition and
        # alignment stay separate so word timestamps never retain attention maps.
        super().__init__(WhisperRecognizer(model, device), aligner, self.capabilities, 30.0, 5.0)


def normalize_whisper_text(result: object) -> str:
    """Validate the plain-text output returned by the Transformers pipeline."""
    if not isinstance(result, dict):
        raise ASROutputError("Whisper returned an unsupported transcription structure")
    text = result.get("text")
    if not isinstance(text, str):
        raise ASROutputError("Whisper response did not contain transcription text")
    return text.strip()
