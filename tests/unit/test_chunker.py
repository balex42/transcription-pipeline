import wave
from pathlib import Path

import numpy as np

from meeting_transcriber.audio.chunker import AudioChunker


def write_wav(path: Path, seconds: float) -> None:
    samples = np.zeros(round(seconds * 16_000), dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples.tobytes())


def test_short_recording_is_one_chunk(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    write_wav(path, 3)
    chunks = AudioChunker(10, 2).chunk(path)
    assert [(chunk.absolute_start, chunk.absolute_end) for chunk in chunks] == [(0.0, 3.0)]


def test_exact_chunk_boundary_does_not_make_empty_tail(tmp_path: Path) -> None:
    path = tmp_path / "exact.wav"
    write_wav(path, 10)
    chunks = AudioChunker(10, 2).chunk(path)
    assert len(chunks) == 1
    assert chunks[0].absolute_end == 10


def test_overlapping_and_final_partial_chunks_have_absolute_offsets(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    write_wav(path, 23)
    chunks = AudioChunker(10, 2).chunk(path)
    assert [(chunk.chunk_id, chunk.absolute_start, chunk.absolute_end) for chunk in chunks] == [
        (0, 0.0, 10.0),
        (1, 8.0, 18.0),
        (2, 16.0, 23.0),
    ]
