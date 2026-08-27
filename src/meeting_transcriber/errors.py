"""Application-specific error types."""


class MeetingTranscriberError(Exception):
    """Base error for expected pipeline failures."""


class AudioProcessingError(MeetingTranscriberError):
    """Raised when ffmpeg or WAV processing fails."""


class ModelLoadError(MeetingTranscriberError):
    """Raised when a local or Hugging Face model cannot be loaded."""


class UnsupportedASRBackendError(MeetingTranscriberError):
    """Raised when an unavailable ASR backend is requested."""


class ASROutputError(MeetingTranscriberError):
    """Raised when a backend cannot normalize its timestamp output."""


class QwenRecognitionError(MeetingTranscriberError):
    """Raised when Qwen ASR cannot produce an internal segment transcript."""


class QwenAlignmentError(MeetingTranscriberError):
    """Raised when Qwen forced alignment cannot produce valid word timing."""


class NemotronStreamingError(MeetingTranscriberError):
    """Raised when Nemotron streaming inference cannot produce valid words."""


class VoxtralStreamingError(MeetingTranscriberError):
    """Raised when Voxtral streaming inference cannot produce valid words."""


class VoxtralTimestampError(MeetingTranscriberError):
    """Raised when Voxtral native emission markers cannot produce timing."""
