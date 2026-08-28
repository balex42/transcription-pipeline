"""Backend-private segmentation helpers for normalized audio."""

from __future__ import annotations

import wave

import numpy as np
from numpy.typing import NDArray

from speech_transcriber.models import AudioSegment, NormalizedAudio


def load_normalized_samples(audio: NormalizedAudio) -> NDArray[np.float32]:
    """Load the project's canonical 16 kHz mono PCM recording."""
    with wave.open(str(audio.path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("normalized audio must be 16-bit mono PCM")
        if wav.getframerate() != audio.metadata.sample_rate:
            raise ValueError("normalized audio sample rate differs from its metadata")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class AudioSegmenter:
    """Create independent overlapping PCM segments for backends that need them."""

    def __init__(self, duration_seconds: float, overlap_seconds: float) -> None:
        if duration_seconds <= 0 or not 0 <= overlap_seconds < duration_seconds:
            raise ValueError(
                "segment overlap must be non-negative and shorter than segment duration"
            )
        self.duration_seconds = duration_seconds
        self.overlap_seconds = overlap_seconds

    def segment(self, audio: NormalizedAudio) -> list[AudioSegment]:
        """Split one normalized recording into offset-aware backend-private segments."""
        samples = load_normalized_samples(audio)
        if not len(samples):
            raise ValueError("normalized audio contains no samples")
        sample_rate = audio.metadata.sample_rate
        segment_samples = round(self.duration_seconds * sample_rate)
        step_samples = segment_samples - round(self.overlap_seconds * sample_rate)
        segments: list[AudioSegment] = []
        start_sample = 0
        while start_sample < len(samples):
            end_sample = min(start_sample + segment_samples, len(samples))
            segments.append(
                AudioSegment(
                    index=len(segments),
                    start=start_sample / sample_rate,
                    end=end_sample / sample_rate,
                    audio=samples[start_sample:end_sample].copy(),
                    sample_rate=sample_rate,
                )
            )
            if end_sample == len(samples):
                break
            start_sample += step_samples
        return segments
