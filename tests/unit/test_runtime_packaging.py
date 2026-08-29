"""Static checks for independent backend runtime dependency boundaries."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def project_dependencies(path: Path) -> list[str]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]["dependencies"]


def test_canary_runtime_uses_byo_torch_and_build_smoke_test() -> None:
    runtime = ROOT / "runtimes/canary/pyproject.toml"
    container = (ROOT / "Containerfile.canary").read_text(encoding="utf-8")

    assert project_dependencies(runtime) == ["nemo-toolkit[asr]==3.0.0"]
    assert "unsafe-best-match" not in container
    assert "--prune torch" in container
    assert "--no-deps --require-hashes" in container
    assert "import nemo, nemo.collections.asr, speech_transcriber, torch" in container
    assert "torch.version.cuda" in container
    assert "faster-whisper" not in runtime.read_text(encoding="utf-8")
    assert "ctranslate2" not in runtime.read_text(encoding="utf-8")


def test_faster_whisper_runtime_keeps_its_existing_stack_isolated() -> None:
    runtime = ROOT / "runtimes/faster-whisper/pyproject.toml"
    container = (ROOT / "Containerfile.faster-whisper").read_text(encoding="utf-8")
    dependencies = project_dependencies(runtime)
    manifest = runtime.read_text(encoding="utf-8")

    assert dependencies == [
        "ctranslate2==4.8.1",
        "faster-whisper==1.2.1",
        "huggingface-hub==1.28.0",
        "numpy==2.2.6",
    ]
    assert "runtimes/faster-whisper/pyproject.toml" in container
    assert "--no-deps --require-hashes" in container
    for package in ("torch", "nemo", "pyannote", "transformers"):
        assert package not in manifest


def test_root_lock_no_longer_owns_dedicated_runtime_dependencies() -> None:
    root_project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    root_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "canary-runtime" not in root_project
    assert "faster-whisper-runtime" not in root_project
    assert "canary-runtime" not in root_lock
    assert "faster-whisper-runtime" not in root_lock


def test_ci_targets_only_the_changed_runtime_or_shared_source() -> None:
    workflow = (ROOT / ".github/workflows/container.yml").read_text(encoding="utf-8")

    assert "runtimes/faster-whisper/**" in workflow
    assert "runtimes/canary/**" in workflow
    assert "needs.changes.outputs.faster_whisper == 'true'" in workflow
    assert "needs.changes.outputs.canary == 'true'" in workflow
