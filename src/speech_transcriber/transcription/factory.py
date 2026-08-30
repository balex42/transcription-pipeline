"""Construction of configured, interchangeable ASR backends.

Backend modules are imported lazily so a runtime image can omit the packages
required by other backends.
"""

from __future__ import annotations

from speech_transcriber.config import DEFAULT_ASR_MODELS, DEFAULT_LANGUAGE, RecognitionConfig
from speech_transcriber.errors import UnsupportedASRBackendError
from speech_transcriber.transcription.base import Transcriber


def create_transcriber(config: RecognitionConfig, device: str, language: str | None) -> Transcriber:
    """Create the selected ASR adapter without loading its model yet.

    ``language`` is the prepared artifact's recording language, passed straight
    through by the recognize CLI. Adapters keep their own normalization of
    that concrete value; language-conditioned adapters may still require one.
    """
    if config.asr_backend not in DEFAULT_ASR_MODELS:
        raise UnsupportedASRBackendError(
            f"Unsupported ASR backend '{config.asr_backend}'. "
            "Supported backends: parakeet, primeline, qwen, nemotron, voxtral, "
            "faster-whisper, canary."
        )
    # Language-conditioned backends require a concrete locale; the prepared
    # artifact is the single source, and its language materializes the
    # adapter-level default only if a caller constructs one without a language.
    conditioned_language = language or DEFAULT_LANGUAGE
    model = config.resolved_asr_model
    if config.asr_backend == "parakeet":
        from speech_transcriber.transcription.parakeet import ParakeetTranscriber

        return ParakeetTranscriber(
            model,
            device,
            config.parakeet_segment_duration,
            config.parakeet_segment_overlap,
        )
    if config.asr_backend == "primeline":
        from speech_transcriber.transcription.primeline import PrimelineTranscriber

        return PrimelineTranscriber(model, device)
    if config.asr_backend == "qwen":
        from speech_transcriber.transcription.qwen import (
            QwenForcedAligner,
            QwenRecognizer,
            QwenTranscriber,
        )

        return QwenTranscriber(
            QwenRecognizer(model, device, conditioned_language),
            QwenForcedAligner(config.qwen_aligner_model, device, conditioned_language),
            config.qwen_segment_duration,
            config.qwen_segment_overlap,
        )
    if config.asr_backend == "nemotron":
        from speech_transcriber.transcription.nemotron import NemotronTranscriber

        return NemotronTranscriber(
            model, device, conditioned_language, config.nemotron_num_lookahead_tokens
        )
    if config.asr_backend == "voxtral":
        from speech_transcriber.transcription.voxtral import VoxtralTranscriber

        return VoxtralTranscriber(
            model,
            device,
            config.voxtral_delay_ms,
            config.voxtral_timestamp_offset_tokens,
        )
    if config.asr_backend == "faster-whisper":
        from speech_transcriber.transcription.faster_whisper import FasterWhisperTranscriber

        return FasterWhisperTranscriber(
            model,
            device,
            language,
            config.faster_whisper_compute_type,
        )
    if config.asr_backend == "canary":
        from speech_transcriber.transcription.canary import CanaryTranscriber

        return CanaryTranscriber(
            model,
            device,
            conditioned_language,
            config.canary_chunk_duration_seconds,
            working_directory=config.working_directory,
        )
    raise AssertionError("validated backend did not match a registered adapter")