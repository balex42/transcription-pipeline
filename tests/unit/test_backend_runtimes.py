"""The authoritative backend -> runtime mapping for worker runtime selection."""

from __future__ import annotations

import pytest

from speech_transcriber.config import ASR_BACKENDS, BACKEND_RUNTIMES


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


def test_no_local_comparison_surface_remains() -> None:
    """Heterogeneous comparisons belong to Argo fan-out, not the Python CLI."""
    import speech_transcriber.config as config_module

    assert not hasattr(config_module, "COMPARE_BACKENDS")
    assert not hasattr(config_module, "TRANSFORMERS_BACKENDS")

    with pytest.raises(ModuleNotFoundError):
        import speech_transcriber.comparison  # noqa: F401