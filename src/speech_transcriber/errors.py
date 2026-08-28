"""Application-specific error types."""


class TranscriberError(Exception):
    """Base error for expected pipeline failures."""


class AudioProcessingError(TranscriberError):
    """Raised when ffmpeg or WAV processing fails."""


class ModelLoadError(TranscriberError):
    """Raised when a local or Hugging Face model cannot be loaded."""


class UnsupportedASRBackendError(TranscriberError):
    """Raised when an unavailable ASR backend is requested."""


class ASROutputError(TranscriberError):
    """Raised when a backend cannot normalize its timestamp output."""


class QwenRecognitionError(TranscriberError):
    """Raised when Qwen ASR cannot produce an internal segment transcript."""


class CohereRecognitionError(TranscriberError):
    """Raised when Cohere ASR cannot produce an internal segment transcript."""


class QwenAlignmentError(TranscriberError):
    """Raised when Qwen forced alignment cannot produce valid word timing."""


class NemotronStreamingError(TranscriberError):
    """Raised when Nemotron streaming inference cannot produce valid words."""


class VoxtralStreamingError(TranscriberError):
    """Raised when Voxtral streaming inference cannot produce valid words."""


class VoxtralTimestampError(TranscriberError):
    """Raised when Voxtral native emission markers cannot produce timing."""
