import json
from pathlib import Path

from speech_transcriber import cli
from speech_transcriber import pipeline as pipeline_module
from speech_transcriber.config import DEFAULT_PYANNOTE_MODEL, DEFAULT_QWEN_ALIGNER_MODEL
from speech_transcriber.models import ASRWord, AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.pipeline import TranscriptionPipeline
from speech_transcriber.prepared import PreparedRecording, sha256_file, write_prepared_recording
from speech_transcriber.transcription.base import TranscriberCapabilities


class FakePreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def normalize(self, _: Path, destination: Path) -> AudioMetadata:
        self.calls += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"normalized audio")
        return AudioMetadata("meeting.wav", 2.0)


class FakeDiarizer:
    def __init__(self) -> None:
        self.calls = 0
        self.released = False

    def diarize(self, _: Path) -> list[DiarizationSegment]:
        self.calls += 1
        return [DiarizationSegment("SPEAKER_00", 0.0, 2.0)]

    def release(self) -> None:
        self.released = True


class FakeTranscriber:
    def __init__(self) -> None:
        self.device = "cpu"
        self.dtype_name = "float32"
        self.model_reference = "fake/asr"
        self.capabilities = TranscriberCapabilities(True, True, True, True)
        self.loaded = False
        self.load_calls = 0
        self.released = False

    def load(self) -> None:
        self.loaded = True
        self.load_calls += 1

    def transcribe(self, _: NormalizedAudio) -> list[ASRWord]:
        return [ASRWord("hallo", 0.5, start=0.0)]

    def release(self) -> None:
        self.released = True


def test_prefetch_qwen_includes_the_forced_aligner(monkeypatch: object) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def prefetch(asr: str, pyannote: str, aligner: str | None) -> None:
        calls.append((asr, pyannote, aligner))

    monkeypatch.setattr(cli, "_prefetch", prefetch)  # type: ignore[attr-defined]

    assert cli.main(["prefetch-models", "--asr", "qwen"]) == 0
    assert calls == [
        ("Qwen/Qwen3-ASR-1.7B-hf", DEFAULT_PYANNOTE_MODEL, DEFAULT_QWEN_ALIGNER_MODEL)
    ]


def test_prefetch_faster_whisper_uses_only_the_model_repository(monkeypatch: object) -> None:
    downloads: list[str] = []

    def snapshot_download(repo: str, **kwargs: object) -> None:
        downloads.append(repo)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    assert cli.main(["prefetch-models", "--asr", "faster-whisper"]) == 0
    assert downloads == [
        "Systran/faster-whisper-large-v3",
        DEFAULT_PYANNOTE_MODEL,
    ]


def test_prefetch_canary_uses_only_its_model_repository(monkeypatch: object) -> None:
    downloads: list[str] = []

    def snapshot_download(repo: str, **kwargs: object) -> None:
        downloads.append(repo)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    assert cli.main(["prefetch-models", "--asr", "canary"]) == 0
    assert downloads == ["nvidia/canary-1b-v2", DEFAULT_PYANNOTE_MODEL]


def test_compare_defaults_to_generic_runtime_backends() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["compare", "input.wav", "--output", "output"])
    assert args.models == "parakeet,primeline,qwen,nemotron,voxtral,granite"


def test_compare_rejects_heterogeneous_backends_before_creating_pipeline(
    monkeypatch: object,
) -> None:
    def unexpected_pipeline(_: object) -> object:
        raise AssertionError("heterogeneous compare must fail before creating the generic pipeline")

    monkeypatch.setattr(cli, "create_default_pipeline", unexpected_pipeline)  # type: ignore[attr-defined]

    assert (
        cli.main(
            [
                "compare",
                "input.wav",
                "--models",
                "faster-whisper,canary",
                "--output",
                "output",
            ]
        )
        == 1
    )


def test_prepare_command_writes_artifact_and_releases_diarizer(
    tmp_path: Path, monkeypatch: object
) -> None:
    preprocessor = FakePreprocessor()
    diarizer = FakeDiarizer()

    def pipeline_factory(config: object) -> TranscriptionPipeline:
        return TranscriptionPipeline(  # type: ignore[arg-type]
            config,
            diarizer_factory=lambda: diarizer,
            transcriber_factory=FakeTranscriber,
            preprocessor=preprocessor,
        )

    monkeypatch.setattr(cli, "create_default_pipeline", pipeline_factory)  # type: ignore[attr-defined]
    output = tmp_path / "prepared"

    assert (
        cli.main(
            [
                "prepare",
                str(tmp_path / "meeting.wav"),
                "--output",
                str(output),
                "--working-directory",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )

    assert (preprocessor.calls, diarizer.calls, diarizer.released) == (1, 1, True)
    assert {path.name for path in output.iterdir()} == {
        "normalized.wav",
        "diarization.json",
        "prepared.json",
    }


def test_transcribe_prepared_uses_only_asr_and_keeps_input_immutable(
    tmp_path: Path, monkeypatch: object
) -> None:
    source = tmp_path / "normalized.wav"
    source.write_bytes(b"normalized audio")
    prepared_directory = tmp_path / "prepared"
    write_prepared_recording(
        PreparedRecording(
            NormalizedAudio(source, AudioMetadata("meeting.wav", 2.0)),
            [DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
            tmp_path,
            normalized_audio_sha256=sha256_file(source),
            diarization_model="pyannote/test",
            language="de-DE",
        ),
        prepared_directory,
    )
    input_manifest = (prepared_directory / "prepared.json").read_bytes()
    transcriber = FakeTranscriber()
    preprocessor = FakePreprocessor()

    def pipeline_factory(config: object) -> TranscriptionPipeline:
        def unexpected_diarizer() -> FakeDiarizer:
            raise AssertionError("transcribe-prepared must not construct a diarizer")

        return TranscriptionPipeline(  # type: ignore[arg-type]
            config,
            diarizer_factory=unexpected_diarizer,
            transcriber_factory=lambda: transcriber,
            preprocessor=preprocessor,
        )

    monkeypatch.setattr(cli, "create_default_pipeline", pipeline_factory)  # type: ignore[attr-defined]
    def unexpected_hash(_: Path) -> str:
        raise AssertionError("recognition must reuse the prepared digest")

    monkeypatch.setattr(pipeline_module, "sha256_file", unexpected_hash)
    output = tmp_path / "result"

    assert (
        cli.main(
            [
                "transcribe-prepared",
                "--prepared",
                str(prepared_directory),
                "--asr",
                "parakeet",
                "--output",
                str(output),
                "--working-directory",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )

    assert preprocessor.calls == 0
    assert transcriber.loaded and transcriber.load_calls == 1 and transcriber.released
    assert (prepared_directory / "prepared.json").read_bytes() == input_manifest
    assert {path.name for path in output.iterdir()} == {
        "transcript.json",
        "transcript.txt",
        "asr_words.json",
        "metadata.json",
    }
    assert json.loads((output / "transcript.json").read_text(encoding="utf-8"))["metadata"][
        "diarization_model"
    ] == "pyannote/test"
    assert json.loads((output / "metadata.json").read_text(encoding="utf-8"))["schema_version"] == 2


def test_recognize_and_finalize_commands_cross_the_artifact_boundary(
    tmp_path: Path, monkeypatch: object
) -> None:
    source = tmp_path / "normalized.wav"
    source.write_bytes(b"normalized audio")
    prepared_directory = tmp_path / "prepared"
    write_prepared_recording(
        PreparedRecording(
            NormalizedAudio(source, AudioMetadata("meeting.wav", 2.0)),
            [DiarizationSegment("SPEAKER_00", 0.0, 2.0)],
            tmp_path,
            normalized_audio_sha256=sha256_file(source),
            diarization_model="pyannote/test",
            language="de-DE",
        ),
        prepared_directory,
    )
    transcriber = FakeTranscriber()

    def recognition_transcriber(config: object, device: str) -> FakeTranscriber:
        return transcriber

    from speech_transcriber.transcription import factory as factory_module

    monkeypatch.setattr(factory_module, "create_transcriber", recognition_transcriber)
    asr = tmp_path / "asr"
    assert (
        cli.main(
            [
                "recognize-prepared",
                "--prepared",
                str(prepared_directory),
                "--asr",
                "parakeet",
                "--output",
                str(asr),
                "--working-directory",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    assert {path.name for path in asr.iterdir()} == {"asr_words.json", "metadata.json"}
    assert transcriber.loaded and transcriber.released

    def unexpected_pipeline(_: object) -> object:
        raise AssertionError("finalization must not create a runtime pipeline")

    monkeypatch.setattr(cli, "create_default_pipeline", unexpected_pipeline)  # type: ignore[attr-defined]
    result = tmp_path / "result"
    assert (
        cli.main(
            [
                "finalize-prepared",
                "--prepared",
                str(prepared_directory),
                "--asr-result",
                str(asr),
                "--expected-backend",
                "parakeet",
                "--output",
                str(result),
                "--working-directory",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    assert {path.name for path in result.iterdir()} == {
        "transcript.json",
        "transcript.txt",
        "asr_words.json",
        "metadata.json",
    }
    assert (result / "metadata.json").read_bytes() == (asr / "metadata.json").read_bytes()


def test_recognition_device_passes_explicit_values_without_torch() -> None:
    assert cli._recognition_device("cuda") == "cuda"
    assert cli._recognition_device("cpu") == "cpu"


def test_memory_metrics_skip_faster_whisper_and_non_cuda_devices() -> None:
    """The torch-free faster-whisper image must never construct Torch metrics."""
    assert cli._memory_metrics("cuda", "faster-whisper") is None
    assert cli._memory_metrics("cpu", "granite") is None


def test_memory_metrics_use_torch_runtime_for_generic_backends(monkeypatch: object) -> None:
    """Torch-bearing backends get CUDA peak accounting through the adapter."""

    class FakeMetrics:
        def __init__(self, device: str) -> None:
            self.device = device

    import speech_transcriber.runtime.device as device_module

    monkeypatch.setattr(  # type: ignore[attr-defined]
        device_module, "TorchMemoryMetrics", FakeMetrics
    )
    metrics = cli._memory_metrics("cuda", "granite")
    assert isinstance(metrics, FakeMetrics)
    assert metrics.device == "cuda"


def test_new_command_parsing() -> None:
    parser = cli.build_parser()
    prepare = parser.parse_args(["prepare", "input.wav", "--output", "/tmp/prepared"])
    transcribe_prepared = parser.parse_args(
        [
            "transcribe-prepared",
            "--prepared",
            "/tmp/prepared",
            "--asr",
            "parakeet",
            "--output",
            "/tmp/result",
        ]
    )
    recognize_prepared = parser.parse_args(
        [
            "recognize-prepared",
            "--prepared",
            "/tmp/prepared",
            "--asr",
            "parakeet",
            "--output",
            "/tmp/asr",
        ]
    )
    finalize_prepared = parser.parse_args(
        [
            "finalize-prepared",
            "--prepared",
            "/tmp/prepared",
            "--asr-result",
            "/tmp/asr",
            "--expected-backend",
            "parakeet",
            "--output",
            "/tmp/result",
        ]
    )

    assert prepare.input == Path("input.wav")
    assert transcribe_prepared.prepared == Path("/tmp/prepared")
    assert recognize_prepared.asr == "parakeet"
    assert finalize_prepared.asr_result == Path("/tmp/asr")
