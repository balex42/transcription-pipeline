"""Application-specific error types."""


class MeetingTranscriberError(Exception):
    """Base error for expected pipeline failures."""


class AudioProcessingError(MeetingTranscriberError):
    """Raised when ffmpeg or WAV processing fails."""


class TimestampParseError(MeetingTranscriberError):
    """Raised when Granite timestamp output is structurally invalid."""


class ModelLoadError(MeetingTranscriberError):
    """Raised when a local or Hugging Face model cannot be loaded."""


class UnsupportedASRBackendError(MeetingTranscriberError):
    """Raised when an unavailable ASR backend is requested."""


class ASROutputError(MeetingTranscriberError):
    """Raised when a backend cannot normalize its timestamp output."""
