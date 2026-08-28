"""Construction of configured, interchangeable ASR backends."""

from __future__ import annotations

from speech_transcriber.config import PipelineConfig
from speech_transcriber.errors import UnsupportedASRBackendError
from speech_transcriber.transcription.base import Transcriber
from speech_transcriber.transcription.nemotron import NemotronTranscriber
from speech_transcriber.transcription.parakeet import ParakeetTranscriber
from speech_transcriber.transcription.qwen import (
    QwenForcedAligner,
    QwenRecognizer,
    QwenTranscriber,
)
from speech_transcriber.transcription.voxtral import VoxtralTranscriber


def create_transcriber(config: PipelineConfig, device: str) -> Transcriber:
    """Create the selected ASR adapter without loading its model yet."""
    if config.asr_backend not in {"parakeet", "qwen", "nemotron", "voxtral"}:
        raise UnsupportedASRBackendError(
            f"Unsupported ASR backend '{config.asr_backend}'. "
            "Supported backends: parakeet, qwen, nemotron, voxtral."
        )
    model = config.resolved_asr_model
    if config.asr_backend == "parakeet":
        return ParakeetTranscriber(
            model,
            device,
            config.parakeet_segment_duration,
            config.parakeet_segment_overlap,
        )
    if config.asr_backend == "qwen":
        return QwenTranscriber(
            QwenRecognizer(model, device, config.language),
            QwenForcedAligner(config.qwen_aligner_model, device, config.language),
            config.qwen_segment_duration,
            config.qwen_segment_overlap,
        )
    if config.asr_backend == "nemotron":
        return NemotronTranscriber(
            model, device, config.language, config.nemotron_num_lookahead_tokens
        )
    if config.asr_backend == "voxtral":
        return VoxtralTranscriber(
            model,
            device,
            config.voxtral_delay_ms,
            config.voxtral_timestamp_offset_tokens,
        )
    raise AssertionError("validated backend did not match a registered adapter")
