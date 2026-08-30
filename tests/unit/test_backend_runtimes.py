"""The authoritative backend -> runtime mapping and local comparison eligibility."""

from __future__ import annotations

import pytest

from speech_transcriber import cli
from speech_transcriber.config import (
    ASR_BACKENDS,
    BACKEND_RUNTIMES,
    COMPARE_BACKENDS,
    TRANSFORMERS_BACKENDS,
)


def test_backend_runtime_mapping() -> None:
    assert BACKEND_RUNTIMES == {
        "parakeet": "nemo",
        "primeline": "nemo",
        "canary": "nemo",
        "qwen": "transformers",
        "nemotron": "transformers",
        "voxtral": "transformers",
        "faster-whisper": "ctranslate2",
    }


def test_mapping_covers_every_public_backend() -> None:
    assert set(BACKEND_RUNTIMES) == set(ASR_BACKENDS)


def test_transformers_backends_drive_local_comparison() -> None:
    assert TRANSFORMERS_BACKENDS == ("qwen", "nemotron", "voxtral")
    assert tuple(COMPARE_BACKENDS) == TRANSFORMERS_BACKENDS


def test_ne_mo_backends_are_not_locally_comparable() -> None:
    for backend in ("parakeet", "primeline", "canary"):
        assert backend not in COMPARE_BACKENDS


def test_ctranslate2_backend_is_not_locally_comparable() -> None:
    assert "faster-whisper" not in COMPARE_BACKENDS


def test_compare_rejects_ne_mo_backends_before_creating_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_pipeline(_: object) -> object:
        raise AssertionError("heterogeneous compare must fail before creating the pipeline")

    monkeypatch.setattr(cli, "create_default_pipeline", unexpected_pipeline)

    assert (
        cli.main(["compare", "input.wav", "--models", "parakeet,voxtral", "--output", "output"])
        == 1
    )


def test_compare_rejects_ctranslate2_backend_before_creating_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_pipeline(_: object) -> object:
        raise AssertionError("heterogeneous compare must fail before creating the pipeline")

    monkeypatch.setattr(cli, "create_default_pipeline", unexpected_pipeline)

    assert (
        cli.main(
            ["compare", "input.wav", "--models", "faster-whisper,voxtral", "--output", "output"]
        )
        == 1
    )


def test_compare_defaults_to_the_transformers_runtime_backends() -> None:
    args = cli.build_parser().parse_args(["compare", "input.wav", "--output", "output"])

    assert args.models == "qwen,nemotron,voxtral"


def test_cli_compare_help_mentions_runtime_semantics() -> None:
    parser = cli.build_parser()
    compare = next(
        action for action in parser._actions if action.dest == "command"  # noqa: SLF001
    ).choices["compare"]
    help_text = next(
        action.help for action in compare._actions if action.dest == "models"  # noqa: SLF001
    )

    assert "Transformers" in help_text
    assert "parakeet" not in help_text