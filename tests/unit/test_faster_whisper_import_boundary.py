"""Prove faster-whisper recognition runs without Torch/Transformers/pyannote.

The dedicated faster-whisper image installs only the faster-whisper-runtime
dependency set. This test runs the real ``recognize-prepared`` CLI path in a
subprocess whose import guard fails if the recognition path imports Torch,
Transformers, pyannote, or unrelated ASR backend modules.
"""

import subprocess
import sys
from pathlib import Path

from speech_transcriber.models import (
    AudioMetadata,
    DiarizationSegment,
    NormalizedAudio,
)
from speech_transcriber.prepared import PreparedRecording, sha256_file, write_prepared_recording


def test_recognize_prepared_faster_whisper_imports_no_ml_stack(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.wav"
    normalized.write_bytes(b"normalized audio")
    prepared = tmp_path / "prepared"
    write_prepared_recording(
        PreparedRecording(
            audio=NormalizedAudio(normalized, AudioMetadata("meeting.wav", 2.0)),
            diarization=[DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
            work_directory=tmp_path,
            normalized_audio_sha256=sha256_file(normalized),
            diarization_model="pyannote/test",
            language="de-DE",
        ),
        prepared,
    )
    asr = tmp_path / "asr"
    command = f"""
import builtins

blocked = (
    "torch", "transformers", "pyannote", "soundfile", "nemo",
    "speech_transcriber.pipeline", "speech_transcriber.diarization",
    "speech_transcriber.audio", "speech_transcriber.runtime",
    "speech_transcriber.transcription.parakeet",
    "speech_transcriber.transcription.qwen",
    "speech_transcriber.transcription.nemotron",
    "speech_transcriber.transcription.voxtral",
    "speech_transcriber.transcription.canary",
)
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise ImportError(f"blocked import: {{name}}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

class Word:
    def __init__(self, word, start, end, probability):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability

class Segment:
    def __init__(self, start, end, text, words):
        self.start = start
        self.end = end
        self.text = text
        self.words = words

class Info:
    language = "de"
    language_probability = 0.98
    duration = 2.0
    duration_after_vad = 2.0

class FakeModel:
    def transcribe(self, audio, **kwargs):
        return (
            [Segment(0.0, 2.0, "Hallo", [Word("Hallo", 0.0, 0.5, 0.99)])],
            Info(),
        )

from speech_transcriber.transcription import faster_whisper as fw
fw._create_whisper_model = lambda model_path, device, compute_type: FakeModel()
from speech_transcriber.cli import main
raise SystemExit(main([
    "recognize-prepared", "--prepared", {str(prepared)!r},
    "--asr", "faster-whisper", "--output", {str(asr)!r},
    "--device", "cuda",
]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in asr.iterdir()} == {"asr_words.json", "metadata.json"}
