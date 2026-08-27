"""Two-phase Qwen adapter hidden behind the common ASR interface."""

from __future__ import annotations

import time
from dataclasses import dataclass

from meeting_transcriber.models import ASRWord, AudioChunk
from meeting_transcriber.runtime.lifecycle import release_model
from meeting_transcriber.transcription.base import TranscriberCapabilities
from meeting_transcriber.transcription.qwen.forced_aligner import QwenForcedAligner
from meeting_transcriber.transcription.qwen.recognizer import QwenRecognizer


@dataclass(frozen=True)
class RecognizedChunk:
    """Qwen ASR text retained until the forced-alignment phase completes."""

    chunk: AudioChunk
    text: str


class QwenTranscriber:
    """Recognize all chunks, release ASR, then align all words with Qwen."""

    capabilities = TranscriberCapabilities(True, True, False, False)

    def __init__(self, recognizer: QwenRecognizer, aligner: QwenForcedAligner) -> None:
        self._recognizer = recognizer
        self._aligner = aligner
        self.model_reference = recognizer.model_reference
        self.device = recognizer.device
        self.dtype_name = recognizer.dtype_name
        self.backend_metrics: dict[str, float] = {}
        self.backend_models = {"qwen_aligner_model": aligner.model_reference}
        self.recognized_chunks: list[RecognizedChunk] = []
        self._recognizer_loaded = False

    def load(self) -> None:
        """Load Qwen ASR once before recognition begins."""
        if self._recognizer_loaded:
            return
        started = time.monotonic()
        self._recognizer.load()
        self._recognizer_loaded = True
        self.backend_metrics["qwen_asr_model_load_seconds"] = time.monotonic() - started

    def transcribe(self, chunk: AudioChunk) -> list[ASRWord]:
        """Support single-chunk callers through the same two-phase lifecycle."""
        self.load()
        return self.transcribe_chunks([chunk])[chunk.chunk_id]

    def transcribe_chunks(self, chunks: list[AudioChunk]) -> dict[int, list[ASRWord]]:
        """Run one ASR pass and one forced-alignment pass across all chunks."""
        if not chunks:
            return {}
        self.load()
        recognized = self._recognize_chunks(chunks)
        self.recognized_chunks = recognized
        if not recognized:
            return {chunk.chunk_id: [] for chunk in chunks}
        return self._align_chunks(chunks, recognized)

    def _recognize_chunks(self, chunks: list[AudioChunk]) -> list[RecognizedChunk]:
        started = time.monotonic()
        try:
            recognized = [
                RecognizedChunk(chunk, text)
                for chunk in chunks
                if (text := self._recognizer.recognize(chunk))
            ]
            self.backend_metrics["qwen_asr_inference_seconds"] = time.monotonic() - started
            return recognized
        finally:
            unload_started = time.monotonic()
            release_model(self._recognizer)
            self._recognizer_loaded = False
            self.backend_metrics["qwen_asr_unload_seconds"] = time.monotonic() - unload_started

    def _align_chunks(
        self, chunks: list[AudioChunk], recognized: list[RecognizedChunk]
    ) -> dict[int, list[ASRWord]]:
        load_started = time.monotonic()
        self._aligner.load()
        self.backend_metrics["qwen_aligner_model_load_seconds"] = time.monotonic() - load_started
        alignment_started = time.monotonic()
        try:
            words_by_chunk: dict[int, list[ASRWord]] = {
                chunk.chunk_id: [] for chunk in chunks
            }
            for recognized_chunk in recognized:
                words_by_chunk[recognized_chunk.chunk.chunk_id] = self._aligner.align(
                    recognized_chunk.chunk, recognized_chunk.text
                )
            self.backend_metrics["qwen_alignment_seconds"] = time.monotonic() - alignment_started
            return words_by_chunk
        finally:
            unload_started = time.monotonic()
            release_model(self._aligner)
            self.backend_metrics["qwen_aligner_unload_seconds"] = time.monotonic() - unload_started

    def release(self) -> None:
        """Release either model if a phase exits early."""
        self._recognizer.release()
        self._aligner.release()
        self._recognizer_loaded = False
