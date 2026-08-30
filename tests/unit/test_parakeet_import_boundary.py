"""Prove Parakeet recognition requires only its NeMo runtime and neutral code.

The real ``recognize`` CLI path runs in a subprocess whose import guard fails
if it imports pyannote, the Transformers ASR stack, CTranslate2, preparation,
or finalization modules.
"""

import subprocess
import sys
import wave
from pathlib import Path

from speech_transcriber.models import AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.prepared import PreparedRecording, sha256_file, write_prepared_recording


def test_recognize_parakeet_imports_no_unrelated_runtime(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.wav"
    with wave.open(str(normalized), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\x00\x00" * 16_000)
    prepared = tmp_path / "prepared"
    write_prepared_recording(
        PreparedRecording(
            audio=NormalizedAudio(normalized, AudioMetadata("meeting.wav", 1.0)),
            diarization=[DiarizationSegment("SPEAKER_00", 0.0, 1.0)],
            work_directory=tmp_path,
            normalized_audio_sha256=sha256_file(normalized),
            diarization_model="pyannote/test",
            language="de-DE",
        ),
        prepared,
    )
    asr = tmp_path / "asr"
    model = tmp_path / "parakeet-tdt-0.6b-v3.nemo"
    model.write_bytes(b"trusted model")
    command = f"""
import builtins

blocked = (
    "pyannote", "transformers", "faster_whisper", "ctranslate2",
    "speech_transcriber.preparation", "speech_transcriber.finalization",
    "speech_transcriber.audio.preprocess",
    "speech_transcriber.transcription.primeline",
    "speech_transcriber.transcription.canary",
    "speech_transcriber.transcription.qwen",
    "speech_transcriber.transcription.nemotron",
    "speech_transcriber.transcription.voxtral",
    "speech_transcriber.transcription.faster_whisper",
)
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise ImportError(f"blocked import: {{name}}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import numpy as np

class Hypothesis:
    timestamp = {{"word": [{{"word": "Hallo!", "start": 0.0, "end": 0.5}}]}}

class FakeModel:
    def transcribe(self, audio_arrays, **kwargs):
        assert isinstance(audio_arrays[0], np.ndarray)
        return [Hypothesis()]

from speech_transcriber.transcription import parakeet
parakeet._restore_parakeet_model = lambda model_path, device: FakeModel()
from speech_transcriber.cli import main
raise SystemExit(main([
    "recognize", "--prepared", {str(prepared)!r},
    "--backend", "parakeet", "--model", {str(model)!r},
    "--output", {str(asr)!r}, "--device", "cuda",
]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in asr.iterdir()} == {"asr_words.json", "metadata.json"}