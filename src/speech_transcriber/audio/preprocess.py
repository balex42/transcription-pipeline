"""ffmpeg-backed input normalization."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from speech_transcriber.errors import AudioProcessingError
from speech_transcriber.models import AudioMetadata

LOGGER = logging.getLogger(__name__)


class AudioPreprocessor:
    """Convert supported recordings to 16 kHz mono 16-bit PCM WAV."""

    def normalize(self, source: Path, destination: Path) -> AudioMetadata:
        """Normalize ``source`` without modifying it and return resulting metadata."""
        if not source.is_file():
            raise AudioProcessingError(f"input recording does not exist: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(destination),
        ]
        LOGGER.info("normalizing audio with ffmpeg")
        self._run(command, "ffmpeg normalization")
        duration = self._duration(destination)
        return AudioMetadata(source=source.name, duration_seconds=duration)

    def _duration(self, path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        output = self._run(command, "ffprobe duration")
        try:
            duration = float(output.stdout.strip())
        except ValueError as error:
            raise AudioProcessingError(
                f"could not determine normalized audio duration for {path}"
            ) from error
        if duration <= 0:
            raise AudioProcessingError(f"normalized audio has non-positive duration: {path}")
        return duration

    @staticmethod
    def _run(command: list[str], operation: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise AudioProcessingError(
                f"{operation} requires ffmpeg and ffprobe on PATH"
            ) from error
        except subprocess.CalledProcessError as error:
            details = error.stderr.strip() or error.stdout.strip() or "no diagnostic output"
            raise AudioProcessingError(f"{operation} failed: {details}") from error
