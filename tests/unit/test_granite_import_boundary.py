"""Prove Granite recognition requires only the generic Transformers runtime."""

import subprocess
import sys
import wave
from pathlib import Path

from speech_transcriber.models import AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.prepared import PreparedRecording, sha256_file, write_prepared_recording


def test_recognize_prepared_granite_imports_no_unrelated_asr_runtime(tmp_path: Path) -> None:
    """End-only words flow through the neutral artifact path without torch."""
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
    model_dir = tmp_path / "granite"
    model_dir.mkdir()
    command = f"""
import builtins, sys

blocked = (
    "faster_whisper", "ctranslate2", "nemo", "speech_transcriber.pipeline",
    "speech_transcriber.diarization",
    "speech_transcriber.transcription.parakeet",
    "speech_transcriber.transcription.qwen", "speech_transcriber.transcription.nemotron",
    "speech_transcriber.transcription.voxtral",
    "speech_transcriber.transcription.canary",
    "speech_transcriber.transcription.primeline",
)
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise ImportError(f"blocked import: {{name}}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

class FakeTokenizer:
    def apply_chat_template(self, conversation, **kwargs):
        content = conversation[0]["content"]
        return content[-1]["text"]

class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert kwargs.get("local_files_only") is True
        assert kwargs.get("trust_remote_code") is False
        return cls()
    def __call__(self, text, audio, **kwargs):
        class Shape:
            def __init__(self):
                self.shape = (1, 5)
            def __len__(self):
                return 5
            def __getitem__(self, index):
                return (1, 5)[index]
        class Inputs(dict):
            def __init__(self):
                super().__init__(input_ids=Shape())
                self.shape = (1, 5)
            def to(self, device, dtype):
                return self
        return Inputs()
    def batch_decode(self, token_ids, **kwargs):
        return ["Hallo [T:45] Welt [T:82]"]

class FakeModel:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert kwargs["local_files_only"] is True
        return cls()
    def to(self, device):
        return self
    def eval(self):
        return self
    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        class Tail:
            def __init__(self):
                self.shape = (1, 30)
        class Seq:
            def __getitem__(self, key):
                return Tail()
            def numel(self):
                return 35
        class Output:
            sequences = Seq()
        return Output()

import speech_transcriber.transcription.granite as granite
granite._transformers_factories = lambda: (FakeProcessor, FakeModel)
from speech_transcriber.cli import main
raise SystemExit(main([
    "recognize-prepared", "--prepared", {str(prepared)!r},
    "--asr", "granite", "--asr-model", {str(model_dir)!r},
    "--output", {str(asr)!r}, "--device", "cpu",
]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in asr.iterdir()} == {"asr_words.json", "metadata.json"}
    import json

    words = json.loads((asr / "asr_words.json").read_text(encoding="utf-8"))
    assert [(word["text"], word["start"], word["end"]) for word in words] == [
        ("Hallo", None, 0.45),
        ("Welt", None, 0.82),
    ]


def test_generic_runtime_does_not_import_transformers_for_granite_registration() -> None:
    """The factory must not import Transformers/Torch/PEFT just to select granite."""
    command = """
import builtins, sys

blocked = ("transformers", "torch", "peft")
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise ImportError(f"blocked import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from pathlib import Path
from speech_transcriber.config import PipelineConfig
from speech_transcriber.transcription.factory import create_transcriber
config = PipelineConfig(Path("in.wav"), Path("out"), Path("work"), asr_backend="granite")
transcriber = create_transcriber(config, "cpu")
assert type(transcriber).__name__ == "GraniteTranscriber"
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")


def test_peft_is_available_with_transformers_in_the_generic_runtime() -> None:
    """Granite's audio LoRA requires peft at import time alongside transformers."""
    command = """
import peft
import transformers
import torch
import speech_transcriber
from transformers.utils import is_peft_available
assert is_peft_available()
print("ok")
"""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")