"""Worker CLI contract: four commands, explicit backend selection, artifact flow."""

import argparse
import json
import logging
from pathlib import Path

import pytest

from speech_transcriber import cli
from speech_transcriber.config import ASR_BACKENDS, DEFAULT_PYANNOTE_MODEL
from speech_transcriber.models import ASRWord, AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.prepared import (
    PreparedRecording,
    load_prepared_recording,
    sha256_file,
    write_prepared_recording,
)
from speech_transcriber.transcription.base import TranscriberCapabilities

PUBLIC_COMMANDS = ("prepare", "recognize", "finalize", "prefetch")
REMOVED_COMMANDS = (
    "transcribe",
    "transcribe-prepared",
    "recognize-prepared",
    "finalize-prepared",
    "compare",
    "prefetch-models",
)


def command_options(command: str) -> set[str]:
    """Return the option strings a worker subcommand accepts."""
    parser = cli.build_parser()
    choices = next(
        action for action in parser._actions if action.dest == "command"  # noqa: SLF001
    ).choices  # type: ignore[attr-defined]
    subparser = choices[command]
    return {option for action in subparser._actions for option in action.option_strings}  # noqa: SLF001


def command_actions(command: str) -> argparse.Namespace:
    """Parse canonical arguments for a worker command and return the namespace."""
    return cli.build_parser()


def write_prepared_fixture(
    tmp_path: Path, language: str = "de-DE"
) -> tuple[Path, Path]:
    """Write a minimal prepared artifact and return (prepared, source)."""
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
            language=language,
        ),
        prepared_directory,
    )
    return prepared_directory, source


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


def test_parser_exposes_exactly_the_four_worker_commands() -> None:
    parser = cli.build_parser()
    subcommands = next(
        action for action in parser._actions if action.dest == "command"  # noqa: SLF001
    ).choices  # type: ignore[attr-defined]

    assert set(subcommands) == set(PUBLIC_COMMANDS)


@pytest.mark.parametrize("command", REMOVED_COMMANDS)
def test_removed_commands_fail_parsing(command: str) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([command])


def test_recognize_requires_prepared_backend_and_output() -> None:
    parser = cli.build_parser()

    for missing in (
        ["recognize", "--backend", "parakeet", "--output", "asr"],
        ["recognize", "--prepared", "prepared", "--output", "asr"],
        ["recognize", "--prepared", "prepared", "--backend", "parakeet"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(missing)


def test_recognize_accepts_no_default_backend() -> None:
    args = cli.build_parser().parse_args(
        ["recognize", "--prepared", "prepared", "--backend", "parakeet", "--output", "asr"]
    )

    assert args.backend == "parakeet"


def test_finalize_requires_prepared_asr_backend_and_output() -> None:
    parser = cli.build_parser()

    for missing in (
        ["finalize", "--asr", "asr", "--backend", "parakeet", "--output", "result"],
        ["finalize", "--prepared", "prepared", "--backend", "parakeet", "--output", "result"],
        ["finalize", "--prepared", "prepared", "--asr", "asr", "--output", "result"],
        ["finalize", "--prepared", "prepared", "--asr", "asr", "--backend", "parakeet"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(missing)


def test_finalize_no_longer_accepts_language() -> None:
    """The prepared artifact owns the language; recognition and finalize have no language flag."""
    for command in ("recognize", "finalize"):
        assert "--language" not in command_options(command)
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([command, "--language", "de-DE"])


def test_prefetch_requires_an_explicit_backend() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["prefetch"])


def test_prepare_exposes_no_backend_specific_recognition_options() -> None:
    # Exact worker surface: input, output, work dir, device, diarization, language, logging.
    assert command_options("prepare") == {
        "-h",
        "--help",
        "--output",
        "--working-directory",
        "--device",
        "--pyannote-model",
        "--language",
        "--num-speakers",
        "--min-speakers",
        "--max-speakers",
        "--log-level",
    }


def test_finalize_exposes_no_recognition_or_diarization_options() -> None:
    # Canonical worker surface: prepared/asr/backend/output plus log-level only.
    assert command_options("finalize") == {
        "-h",
        "--help",
        "--prepared",
        "--asr",
        "--backend",
        "--output",
        "--log-level",
    }


@pytest.mark.parametrize("backend", ASR_BACKENDS)
def test_recognize_supports_every_public_backend(backend: str) -> None:
    args = cli.build_parser().parse_args(
        ["recognize", "--prepared", "prepared", "--backend", backend, "--output", "asr"]
    )

    assert args.backend == backend


def test_recognize_exposes_no_language_flag_for_artifact_inheritance() -> None:
    """The prepared artifact is the immutable language source; there is no override."""
    args = cli.build_parser().parse_args(
        ["recognize", "--prepared", "prepared", "--backend", "parakeet", "--output", "asr"]
    )

    assert not hasattr(args, "language")
    assert "--language" not in command_options("recognize")


@pytest.mark.parametrize(
    ("backend", "option", "value"),
    [
        ("parakeet", "--voxtral-delay-ms", "480"),
        ("primeline", "--parakeet-segment-duration", "120"),
        ("qwen", "--canary-chunk-duration", "20"),
        ("canary", "--qwen-segment-duration", "120"),
        ("faster-whisper", "--nemotron-num-lookahead-tokens", "0"),
    ],
)
def test_recognize_rejects_explicit_options_for_another_backend(
    backend: str, option: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "recognize",
                "--prepared",
                "prepared",
                "--backend",
                backend,
                "--output",
                "asr",
                option,
                value,
            ]
        )

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert option in stderr
    assert backend in stderr
    assert "does not apply" in stderr


@pytest.mark.parametrize(
    ("backend", "options"),
    [
        ("parakeet", ["--parakeet-segment-duration", "120"]),
        ("primeline", []),
        ("qwen", ["--qwen-segment-overlap", "10"]),
        ("nemotron", ["--nemotron-num-lookahead-tokens", "0"]),
        ("voxtral", ["--voxtral-delay-ms", "480"]),
        ("faster-whisper", ["--faster-whisper-compute-type", "int8_float16"]),
        ("canary", ["--canary-chunk-duration", "20"]),
    ],
)
def test_recognize_accepts_options_owned_by_the_selected_backend(
    backend: str, options: list[str]
) -> None:
    args = cli.build_parser().parse_args(
        ["recognize", "--prepared", "prepared", "--backend", backend, "--output", "asr"]
        + options
    )

    cli._validate_backend_cli_options(args)


def test_prepare_command_writes_artifact_and_releases_diarizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preprocessor = FakePreprocessor()
    diarizer = FakeDiarizer()

    def runner_factory(config: object) -> object:
        from speech_transcriber.preparation import PreparationRunner

        return PreparationRunner(
            config,  # type: ignore[arg-type]
            diarizer_factory=lambda: diarizer,
            preprocessor=preprocessor,
        )

    monkeypatch.setattr(
        "speech_transcriber.preparation.PreparationRunner.create_default", runner_factory
    )
    monkeypatch.setenv("PYANNOTE_MODEL", "/models/pyannote-community-1")
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
    assert load_prepared_recording(output).diarization_model == "/models/pyannote-community-1"


@pytest.mark.parametrize("prepared_language", ["fr-FR", "en-US"])
def test_recognize_passes_the_prepared_language_to_the_backend_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_language: str,
) -> None:
    """The prepared artifact's language reaches the adapter with no override path."""
    prepared_directory, _source = write_prepared_fixture(tmp_path, language=prepared_language)
    seen: dict[str, object] = {}

    def fake_transcriber(
        config: object, device: str, language: str
    ) -> FakeTranscriber:
        seen["language"] = language
        return FakeTranscriber()

    import speech_transcriber.transcription.factory as factory_module

    monkeypatch.setattr(factory_module, "create_transcriber", fake_transcriber)
    monkeypatch.setenv("LANGUAGE", "de-DE")

    assert (
        cli.main(
            [
                "recognize",
                "--prepared",
                str(prepared_directory),
                "--backend",
                "parakeet",
                "--output",
                str(tmp_path / "asr"),
                "--working-directory",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    assert seen["language"] == prepared_language


def test_recognize_writes_canonical_asr_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared_directory, _source = write_prepared_fixture(tmp_path)
    transcriber = FakeTranscriber()

    def fake_transcriber(config: object, device: str, language: str) -> FakeTranscriber:
        assert device == "cpu"
        assert language == "de-DE"
        return transcriber

    import speech_transcriber.transcription.factory as factory_module

    monkeypatch.setattr(factory_module, "create_transcriber", fake_transcriber)
    asr = tmp_path / "asr"

    assert (
        cli.main(
            [
                "recognize",
                "--prepared",
                str(prepared_directory),
                "--backend",
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


def test_recognize_wires_memory_metrics_from_the_backend_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared_directory, _source = write_prepared_fixture(tmp_path)
    seen: dict[str, object] = {}

    def fake_transcriber(config: object, device: str, language: str) -> FakeTranscriber:
        return FakeTranscriber()

    def fake_runner(metrics: object) -> object:
        seen["metrics"] = metrics

        class FakeRunner:
            def recognize(self, *_: object) -> None:
                raise AssertionError("not reached")

        return FakeRunner()

    import speech_transcriber.recognition as recognition_module
    import speech_transcriber.transcription.factory as factory_module

    monkeypatch.setattr(factory_module, "create_transcriber", fake_transcriber)
    monkeypatch.setattr(recognition_module, "RecognitionRunner", fake_runner)
    monkeypatch.setattr(cli, "_memory_metrics", lambda device, backend: "metrics-for-parakeet")

    with pytest.raises(AssertionError, match="not reached"):
        cli.main(
            [
                "recognize",
                "--prepared",
                str(prepared_directory),
                "--backend",
                "parakeet",
                "--output",
                str(tmp_path / "asr"),
            ]
        )

    assert seen["metrics"] == "metrics-for-parakeet"


def test_recognize_and_finalize_commands_cross_the_artifact_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared_directory, _source = write_prepared_fixture(tmp_path)
    transcriber = FakeTranscriber()

    def recognition_transcriber(
        config: object, device: str, language: str
    ) -> FakeTranscriber:
        return transcriber

    import speech_transcriber.transcription.factory as factory_module

    monkeypatch.setattr(factory_module, "create_transcriber", recognition_transcriber)
    asr = tmp_path / "asr"
    assert (
        cli.main(
            [
                "recognize",
                "--prepared",
                str(prepared_directory),
                "--backend",
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

    def unexpected_transcriber(config: object, device: str, language: str) -> object:
        raise AssertionError("finalization must not create a backend adapter")

    monkeypatch.setattr(factory_module, "create_transcriber", unexpected_transcriber)
    result = tmp_path / "result"
    assert (
        cli.main(
            [
                "finalize",
                "--prepared",
                str(prepared_directory),
                "--asr",
                str(asr),
                "--backend",
                "parakeet",
                "--output",
                str(result),
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


def test_finalize_inherits_the_prepared_language_into_the_transcript(
    tmp_path: Path,
) -> None:
    prepared_directory, _source = write_prepared_fixture(tmp_path, language="fr-FR")
    asr = tmp_path / "asr"
    asr.mkdir()
    (asr / "asr_words.json").write_text(
        json.dumps(
            [
                {"text": "bonjour", "start": 0.0, "end": 0.5, "confidence": 0.9},
                {"text": "monde", "start": 0.5, "end": 1.5, "confidence": 0.9},
            ]
        ),
        encoding="utf-8",
    )
    (asr / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "asr_words_file": "asr_words.json",
                "backend": "parakeet",
                "model": "parakeet",
                "device": "cuda",
                "dtype": "float16",
                "audio_duration_seconds": 2.0,
                "model_load_seconds": 1.0,
                "transcription_seconds": 0.5,
                "total_asr_seconds": 1.5,
                "real_time_factor": 0.75,
                "normalized_audio_sha256": sha256_file(_source),
                "runtime": {"name": "nemo", "version": "3.0.0", "components": {}},
            }
        ),
        encoding="utf-8",
    )
    result = tmp_path / "result"

    assert (
        cli.main(
            [
                "finalize",
                "--prepared",
                str(prepared_directory),
                "--asr",
                str(asr),
                "--backend",
                "parakeet",
                "--output",
                str(result),
            ]
        )
        == 0
    )

    transcript = json.loads((result / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["metadata"]["language"] == "fr-FR"


def test_finalize_rejects_a_mismatched_backend(tmp_path: Path) -> None:
    prepared_directory, _source = write_prepared_fixture(tmp_path)
    asr = tmp_path / "asr"
    asr.mkdir()
    (asr / "asr_words.json").write_text("[]", encoding="utf-8")
    (asr / "metadata.json").write_text(
        json.dumps({"schema_version": 2, "backend": "qwen"}), encoding="utf-8"
    )

    assert (
        cli.main(
            [
                "finalize",
                "--prepared",
                str(prepared_directory),
                "--asr",
                str(asr),
                "--backend",
                "canary",
                "--output",
                str(tmp_path / "result"),
            ]
        )
        == 1
    )


def test_recognition_device_passes_explicit_values_without_torch() -> None:
    assert cli._recognition_device("cuda") == "cuda"
    assert cli._recognition_device("cpu") == "cpu"


def test_memory_metrics_skip_the_ctranslate2_runtime_and_non_cuda_devices() -> None:
    """The torch-free CTranslate2 image must never construct Torch metrics."""
    assert cli._memory_metrics("cuda", "faster-whisper") is None
    assert cli._memory_metrics("cpu", "parakeet") is None


def test_memory_metrics_use_torch_runtime_for_other_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch-bearing runtimes get CUDA peak accounting through the adapter."""

    class FakeMetrics:
        def __init__(self, device: str) -> None:
            self.device = device

    import speech_transcriber.runtime.device as device_module

    monkeypatch.setattr(device_module, "TorchMemoryMetrics", FakeMetrics)
    metrics = cli._memory_metrics("cuda", "parakeet")
    assert isinstance(metrics, FakeMetrics)
    assert metrics.device == "cuda"


@pytest.mark.parametrize(
    ("environment", "arguments", "expected"),
    [
        ({}, [], logging.INFO),
        ({"LOG_LEVEL": "DEBUG"}, [], logging.DEBUG),
        ({"LOG_LEVEL": "DEBUG"}, ["--log-level", "ERROR"], logging.ERROR),
    ],
)
def test_logging_precedence_is_cli_then_environment_then_info(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    arguments: list[str],
    expected: int,
) -> None:
    configured: list[int] = []
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        cli.logging, "basicConfig", lambda **values: configured.append(values["level"])
    )
    monkeypatch.setattr(cli, "_prefetch", lambda *_: None)

    assert cli.main(["prefetch", "--backend", "parakeet", *arguments]) == 0
    assert configured == [expected]


def test_invalid_environment_log_level_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "banana")

    with pytest.raises(SystemExit) as error:
        cli.main(["prefetch", "--backend", "parakeet"])

    assert error.value.code == 2
    assert "LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR" in capsys.readouterr().err


def test_prefetch_qwen_includes_the_forced_aligner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def prefetch(
        backend: str,
        model: str | None,
        qwen_aligner_model: str,
        pyannote_model: str,
    ) -> None:
        calls.append((backend, model, qwen_aligner_model, pyannote_model))

    monkeypatch.setattr(cli, "_prefetch", prefetch)

    assert cli.main(["prefetch", "--backend", "qwen"]) == 0
    assert calls == [("qwen", None, "Qwen/Qwen3-ForcedAligner-0.6B-hf", DEFAULT_PYANNOTE_MODEL)]


def test_prefetch_faster_whisper_uses_only_the_model_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[str] = []

    def snapshot_download(repo: str, **kwargs: object) -> None:
        downloads.append(repo)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    assert cli.main(["prefetch", "--backend", "faster-whisper"]) == 0
    assert downloads == [
        "Systran/faster-whisper-large-v3",
        DEFAULT_PYANNOTE_MODEL,
    ]


def test_prefetch_canary_uses_only_its_model_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[str] = []

    def snapshot_download(repo: str, **kwargs: object) -> None:
        downloads.append(repo)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    assert cli.main(["prefetch", "--backend", "canary"]) == 0
    assert downloads == ["nvidia/canary-1b-v2", DEFAULT_PYANNOTE_MODEL]
