"""Prove Canary recognition requires only its NeMo runtime and neutral code."""

import subprocess
import sys
from pathlib import Path

from speech_transcriber.models import AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.prepared import PreparedRecording, sha256_file, write_prepared_recording


def test_recognize_prepared_canary_imports_no_unrelated_asr_runtime(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.wav"
    normalized.write_bytes(b"normalized audio")
    prepared = tmp_path / "prepared"
    write_prepared_recording(
        PreparedRecording(
            audio=NormalizedAudio(normalized, AudioMetadata("meeting.wav", 45.0)),
            diarization=[DiarizationSegment("SPEAKER_00", 0.0, 45.0)],
            work_directory=tmp_path,
            normalized_audio_sha256=sha256_file(normalized),
            diarization_model="pyannote/test",
            language="de-DE",
        ),
        prepared,
    )
    asr = tmp_path / "asr"
    model = tmp_path / "canary-1b-v2.nemo"
    model.write_bytes(b"trusted model")
    command = f"""
import builtins

blocked = (
    "faster_whisper", "ctranslate2", "speech_transcriber.pipeline",
    "speech_transcriber.audio", "speech_transcriber.diarization",
    "speech_transcriber.runtime", "speech_transcriber.transcription.parakeet",
    "speech_transcriber.transcription.qwen", "speech_transcriber.transcription.nemotron",
    "speech_transcriber.transcription.voxtral",
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
            "source_lang": "de",
            "target_lang": "de",
            "timestamps": True,
        }}
        return [Hypothesis()]

from speech_transcriber.transcription import canary
canary._restore_canary_model = lambda model_path, device: FakeModel()
from speech_transcriber.cli import main
raise SystemExit(main([
    "recognize-prepared", "--prepared", {str(prepared)!r},
    "--asr", "canary", "--asr-model", {str(model)!r},
    "--output", {str(asr)!r}, "--device", "cuda",
]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in asr.iterdir()} == {"asr_words.json", "metadata.json"}
