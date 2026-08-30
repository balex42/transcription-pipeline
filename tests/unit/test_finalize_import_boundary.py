import subprocess
import sys
from pathlib import Path

from speech_transcriber.asr_artifact import write_asr_recognition
from speech_transcriber.models import (
    ASRRecognitionResult,
    ASRRunMetadata,
    ASRWord,
    AudioMetadata,
    DiarizationSegment,
    NormalizedAudio,
)
from speech_transcriber.prepared import PreparedRecording, sha256_file, write_prepared_recording


def test_finalize_command_imports_no_ml_or_backend_modules(tmp_path: Path) -> None:
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
    recognition = ASRRecognitionResult(
        words=[ASRWord("hallo", end=0.5, start=0.0)],
        metadata=ASRRunMetadata(
            backend="faster-whisper",
            model="Systran/faster-whisper-large-v3",
            device="cuda",
            dtype="float16",
            audio_duration_seconds=2.0,
            model_load_seconds=1.0,
            transcription_seconds=0.5,
            total_asr_seconds=1.5,
            real_time_factor=0.75,
            peak_cuda_memory_allocated_bytes=None,
            peak_cuda_memory_reserved_bytes=None,
            normalized_audio_sha256=sha256_file(normalized),
        ),
    )
    write_asr_recognition(recognition, asr)
    result = tmp_path / "result"
    command = f"""
import builtins

blocked = (
    "numpy", "torch", "transformers", "pyannote", "soundfile", "nemo",
    "speech_transcriber.pipeline", "speech_transcriber.transcription",
    "speech_transcriber.diarization", "speech_transcriber.audio", "speech_transcriber.runtime",
)
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise ImportError(f"blocked import: {{name}}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from speech_transcriber.cli import main
raise SystemExit(main([
    "finalize", "--prepared", {str(prepared)!r},
    "--asr", {str(asr)!r}, "--backend", "faster-whisper",
    "--output", {str(result)!r},
]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert (result / "transcript.json").is_file()
