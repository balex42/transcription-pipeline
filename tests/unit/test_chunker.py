import wave
from pathlib import Path

import numpy as np

from meeting_transcriber.audio.segmenter import AudioSegmenter
from meeting_transcriber.models import AudioMetadata, NormalizedAudio


def write_wav(path: Path, seconds: float) -> None:
    samples = np.zeros(round(seconds * 16_000), dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples.tobytes())


def audio(path: Path, seconds: float) -> NormalizedAudio:
    write_wav(path, seconds)
    return NormalizedAudio(path, AudioMetadata(path.name, seconds))


def test_short_recording_is_one_segment(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    segments = AudioSegmenter(10, 2).segment(audio(path, 3))
    assert [(segment.start, segment.end) for segment in segments] == [(0.0, 3.0)]


def test_exact_segment_boundary_does_not_make_empty_tail(tmp_path: Path) -> None:
    path = tmp_path / "exact.wav"
    segments = AudioSegmenter(10, 2).segment(audio(path, 10))
    assert len(segments) == 1
    assert segments[0].end == 10


def test_overlapping_and_final_partial_segments_have_absolute_offsets(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    segments = AudioSegmenter(10, 2).segment(audio(path, 23))
    assert [(segment.index, segment.start, segment.end) for segment in segments] == [
        (0, 0.0, 10.0),
        (1, 8.0, 18.0),
        (2, 16.0, 23.0),
    ]
