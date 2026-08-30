"""Prove Primeline recognition requires only the NeMo runtime and neutral code."""

import subprocess
import sys
import wave
from pathlib import Path

from speech_transcriber.models import AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.prepared import PreparedRecording, sha256_file, write_prepared_recording


def test_recognize_prepared_primeline_imports_no_unrelated_asr_runtime(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.wav"
    with wave.open(str(normalized), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 16000)
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
    model = tmp_path / "2_95_WER.nemo"
    model.write_bytes(b"trusted model")
    command = f"""
import builtins

blocked = (
    "transformers", "faster_whisper", "ctranslate2", "speech_transcriber.pipeline",
    "speech_transcriber.audio", "speech_transcriber.diarization",
    "speech_transcriber.runtime", "speech_transcriber.transcription.parakeet",
    "speech_transcriber.transcription.qwen", "speech_transcriber.transcription.nemotron",
    "speech_transcriber.transcription.voxtral",
    "speech_transcriber.transcription.canary",
)
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise ImportError(f"blocked import: {{name}}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

class Hypothesis:
    timestamp = {{"word": [{{"word": "Hallo!", "start": 0.0, "end": 0.5}}]}}

class FakeModel:
    def transcribe(self, audio, **kwargs):
        assert kwargs == {{
            "batch_size": 1,
            "return_hypotheses": True,
            "timestamps": True,
        }}
        return [Hypothesis()]

from speech_transcriber.transcription import primeline
primeline._restore_primeline_model = lambda model_path, device: FakeModel()
from speech_transcriber.cli import main
raise SystemExit(main([
    "recognize-prepared", "--prepared", {str(prepared)!r},
    "--asr", "primeline", "--asr-model", {str(model)!r},
    "--output", {str(asr)!r}, "--device", "cuda",
]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in asr.iterdir()} == {"asr_words.json", "metadata.json"}


def test_transformers_runtime_does_not_import_nemo_for_primeline_registration() -> None:
    """The factory must not need NeMo just because the Primeline backend exists."""
    command = """
import builtins, sys

blocked = ("nemo", "torch")
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise ImportError(f"blocked import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from pathlib import Path
from speech_transcriber.config import PipelineConfig
from speech_transcriber.transcription.factory import create_transcriber
config = PipelineConfig(Path("in.wav"), Path("out"), Path("work"), asr_backend="primeline")
transcriber = create_transcriber(config, "cpu")
assert type(transcriber).__name__ == "PrimelineTranscriber"
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")