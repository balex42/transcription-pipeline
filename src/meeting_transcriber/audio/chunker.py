"""Deterministic fixed-window PCM audio chunking."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from meeting_transcriber.models import AudioChunk


class AudioChunker:
    """Split normalized 16-bit mono WAV audio into overlapping chunks."""

    def __init__(self, duration_seconds: float = 180.0, overlap_seconds: float = 15.0) -> None:
        if duration_seconds <= 0 or not 0 <= overlap_seconds < duration_seconds:
            raise ValueError(
                "overlap_seconds must be non-negative and shorter than duration_seconds"
            )
        self.duration_seconds = duration_seconds
        self.overlap_seconds = overlap_seconds

    def chunk(self, normalized_wav: Path) -> list[AudioChunk]:
        """Read a normalized WAV once and return chunks with absolute offsets."""
        samples, sample_rate = self._read_pcm(normalized_wav)
        total_samples = len(samples)
        if total_samples == 0:
            raise ValueError("normalized WAV contains no audio samples")
        chunk_samples = round(self.duration_seconds * sample_rate)
        overlap_samples = round(self.overlap_seconds * sample_rate)
        step_samples = chunk_samples - overlap_samples
        chunks: list[AudioChunk] = []
        start_sample = 0
        chunk_id = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunks.append(
                AudioChunk(
                    chunk_id=chunk_id,
                    absolute_start=start_sample / sample_rate,
                    absolute_end=end_sample / sample_rate,
                    audio=samples[start_sample:end_sample].copy(),
                    sample_rate=sample_rate,
                )
            )
            if end_sample == total_samples:
                break
            start_sample += step_samples
            chunk_id += 1
        return chunks

    @staticmethod
    def _read_pcm(path: Path) -> tuple[NDArray[np.float32], int]:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("AudioChunker requires 16-bit mono normalized WAV input")
            sample_rate = wav.getframerate()
            if sample_rate != 16_000:
                raise ValueError("AudioChunker requires 16 kHz normalized WAV input")
            raw = wav.readframes(wav.getnframes())
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return samples, sample_rate
