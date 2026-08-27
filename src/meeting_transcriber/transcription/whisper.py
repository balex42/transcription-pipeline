"""Native Transformers pipeline adapter for Whisper large-v3."""

from __future__ import annotations

from typing import Any, Protocol, cast

from meeting_transcriber.audio.segmenter import AudioSegmenter
from meeting_transcriber.errors import ASROutputError, ModelLoadError
from meeting_transcriber.models import ASRWord, AudioSegment, NormalizedAudio
from meeting_transcriber.runtime.device import inference_dtype
from meeting_transcriber.transcription.base import Transcriber, TranscriberCapabilities
from meeting_transcriber.transcription.segments import reconcile_segment_words


class _WhisperModel(Protocol):
    def to(self, device: str) -> object: ...

    def eval(self) -> object: ...


class _WhisperPipeline(Protocol):
    def __call__(self, inputs: object, **kwargs: object) -> dict[str, object]: ...


class WhisperTranscriber(Transcriber):
    """Use Transformers Whisper ASR pipeline with German word timestamps."""

    capabilities = TranscriberCapabilities(True, True, True, True)

    def __init__(self, model: str, device: str) -> None:
        self.model_reference = model
        self.device = device
        _, self.dtype_name = inference_dtype(device)
        self._model: _WhisperModel | None = None
        self._pipeline: _WhisperPipeline | None = None
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        # Pipeline calls must never exceed the model's 30-second window: longer
        # chunks are silently truncated by the feature extractor. One pipeline
        # call per segment also bounds the cross-attention tensors that
        # word-level DTW timestamps otherwise accumulate across the meeting.
        self._segmenter = AudioSegmenter(30.0, 5.0)
        self.backend_configuration = {
            "segment_duration_seconds": 30.0,
            "segment_overlap_seconds": 5.0,
        }

    def load(self) -> None:
        """Load Whisper and its ASR pipeline lazily."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Transcribe segment-sized windows sequentially and reconcile overlaps."""
        segments = self._segmenter.segment(audio)
        self.backend_metrics["segments_processed"] = float(len(segments))
        return reconcile_segment_words(
            segments, {segment.index: self._transcribe_segment(segment) for segment in segments}
        )

    def _transcribe_segment(self, segment: AudioSegment) -> list[ASRWord]:
        """Return word timestamps relative to one model-window-sized segment."""
        pipeline = self._load()
        import torch

        with torch.inference_mode():
            result = pipeline(
                segment.audio,
                return_timestamps="word",
                generate_kwargs={"language": "german", "task": "transcribe"},
            )
        return normalize_whisper_chunks(result.get("chunks", []))

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
        """Drop model and pipeline references for sequential GPU execution."""
        self._model = None
        self._pipeline = None


def normalize_whisper_chunks(chunks: object) -> list[ASRWord]:
    """Normalize Transformers pipeline ``return_timestamps='word'`` output."""
    if not isinstance(chunks, list):
        raise ASROutputError("Whisper returned an unsupported word timestamp structure")
    words: list[ASRWord] = []
    for item in chunks:
        if not isinstance(item, dict):
            raise ASROutputError("Whisper returned a non-object word timestamp")
        text = str(item.get("text", "")).strip()
        timestamp = item.get("timestamp")
        if not text:
            continue
        if not isinstance(timestamp, tuple | list) or len(timestamp) != 2:
            raise ASROutputError("Whisper word timestamp is missing start/end values")
        start, end = timestamp
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ASROutputError("Whisper word timestamp must contain numeric values")
        if end < start:
            raise ASROutputError("Whisper word timestamp ends before it starts")
        words.append(ASRWord(text=text, start=float(start), end=float(end)))
    return words
