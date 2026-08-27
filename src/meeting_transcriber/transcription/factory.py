"""Construction of configured, interchangeable ASR backends."""

from __future__ import annotations

from meeting_transcriber.config import PipelineConfig
from meeting_transcriber.errors import UnsupportedASRBackendError
from meeting_transcriber.transcription.base import Transcriber
from meeting_transcriber.transcription.parakeet import ParakeetTranscriber
from meeting_transcriber.transcription.qwen import (
    QwenForcedAligner,
    QwenRecognizer,
    QwenTranscriber,
)
from meeting_transcriber.transcription.whisper import WhisperTranscriber


def create_transcriber(config: PipelineConfig, device: str) -> Transcriber:
    """Create the selected ASR adapter without loading its model yet."""
    if config.asr_backend not in {"parakeet", "whisper", "qwen"}:
        raise UnsupportedASRBackendError(
            f"Unsupported ASR backend '{config.asr_backend}'. "
            "Supported backends: parakeet, whisper, qwen."
        )
    model = config.resolved_asr_model
    if config.asr_backend == "parakeet":
        return ParakeetTranscriber(model, device)
    if config.asr_backend == "whisper":
        return WhisperTranscriber(model, device)
    if config.asr_backend == "qwen":
        return QwenTranscriber(
            QwenRecognizer(model, device),
            QwenForcedAligner(config.qwen_aligner_model, device),
        )
    raise AssertionError("validated backend did not match a registered adapter")
