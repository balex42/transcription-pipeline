"""Static checks for the runtime-oriented container and CI naming."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def project_dependencies(path: str) -> list[str]:
    data = tomllib.loads(read(path))
    return data["project"]["dependencies"]


def test_containerfiles_are_runtime_named() -> None:
    assert (ROOT / "Containerfile.transformers").is_file()
    assert (ROOT / "Containerfile.nemo").is_file()
    assert (ROOT / "Containerfile.ctranslate2").is_file()
    for stale in ("Containerfile", "Containerfile.canary", "Containerfile.faster-whisper"):
        assert not (ROOT / stale).exists()


def test_runtime_directories_are_runtime_named() -> None:
    assert (ROOT / "runtimes/nemo/pyproject.toml").is_file()
    assert (ROOT / "runtimes/ctranslate2/pyproject.toml").is_file()
    assert not (ROOT / "runtimes/canary").exists()
    assert not (ROOT / "runtimes/faster-whisper").exists()


def test_nemo_runtime_project_describes_the_generic_ne_mo_runtime() -> None:
    manifest = tomllib.loads(read("runtimes/nemo/pyproject.toml"))
    lock = read("runtimes/nemo/uv.lock")

    assert manifest["project"]["name"] == "speech-transcriber-nemo-runtime"
    assert manifest["project"]["description"] == (
        "Locked third-party dependencies for the NeMo ASR recognition runtime"
    )
    assert project_dependencies("runtimes/nemo/pyproject.toml") == ["nemo-toolkit[asr]==3.0.0"]
    assert "speech-transcriber-nemo-runtime" in lock
    assert "canary-runtime" not in lock
    assert "faster-whisper" not in read("runtimes/nemo/pyproject.toml")
    assert "ctranslate2" not in read("runtimes/nemo/pyproject.toml")


def test_ctranslate2_runtime_project_describes_the_runtime() -> None:
    manifest = tomllib.loads(read("runtimes/ctranslate2/pyproject.toml"))
    lock = read("runtimes/ctranslate2/uv.lock")

    assert manifest["project"]["name"] == "speech-transcriber-ctranslate2-runtime"
    assert "CTranslate2 recognition runtime" in manifest["project"]["description"]
    assert project_dependencies("runtimes/ctranslate2/pyproject.toml") == [
        "ctranslate2==4.8.1",
        "faster-whisper==1.2.1",
        "huggingface-hub==1.28.0",
        "numpy==2.2.6",
    ]
    assert "speech-transcriber-ctranslate2-runtime" in lock
    assert "faster-whisper-runtime" not in lock


def test_nemo_image_is_generic_and_supports_all_three_adapters() -> None:
    container = read("Containerfile.nemo")

    assert "runtimes/nemo/pyproject.toml" in container
    assert "import os, pwd, nemo, nemo.collections.asr, speech_transcriber" in container
    for backend in ("parakeet", "primeline", "canary"):
        assert f"speech_transcriber.transcription.{backend}" in container
    assert "unsafe-best-match" not in container
    assert "--prune torch" in container
    assert "--no-deps --require-hashes" in container
    assert "pwd.getpwuid(os.getuid())" in container
    assert "torch.version.cuda" in container


def test_ctranslate2_image_describes_the_runtime_and_stays_isolated() -> None:
    container = read("Containerfile.ctranslate2")
    manifest = read("runtimes/ctranslate2/pyproject.toml")

    assert "runtimes/ctranslate2/pyproject.toml" in container
    assert "CTranslate2" in container
    assert "--no-deps --require-hashes" in container
    for package in ("torch", "nemo", "pyannote", "transformers"):
        assert package not in manifest


def test_transformers_image_supports_prepare_and_its_backends() -> None:
    container = read("Containerfile.transformers")

    assert "--extra transformers" in container
    # No NeMo or CTranslate2 stack is installed in the Transformers image.
    assert "nemo-toolkit" not in container
    assert "ctranslate2==" not in container.lower().replace("containerfile.ctranslate2", "")


def test_root_lock_owns_the_transformers_extra_not_a_generic_runtime_extra() -> None:
    manifest = tomllib.loads(read("pyproject.toml"))
    lock = read("uv.lock")

    assert set(manifest["project"]["optional-dependencies"]) == {"transformers", "dev"}
    assert 'provides-extras = ["transformers", "dev"]' in lock


def test_ci_uses_runtime_jobs_filters_and_images() -> None:
    workflow = read(".github/workflows/container.yml")

    for job in ("build-transformers", "build-nemo", "build-ctranslate2"):
        assert job in workflow
    for output in ("transformers", "nemo", "ctranslate2"):
        assert f"needs.changes.outputs.{output} == 'true'" in workflow
    for containerfile in (
        "Containerfile.transformers",
        "Containerfile.nemo",
        "Containerfile.ctranslate2",
    ):
        assert f"file: {containerfile}" in workflow
    for suffix in ("-transformers", "-nemo", "-ctranslate2"):
        assert f"${{{{ env.IMAGE_NAME }}}}{suffix}" in workflow.replace("}}{{", "}}{")
    assert "runtimes/nemo/**" in workflow
    assert "runtimes/ctranslate2/**" in workflow


def test_ci_has_no_backend_named_container_references() -> None:
    workflow = read(".github/workflows/container.yml")

    assert "Containerfile.canary" not in workflow
    assert "Containerfile.faster-whisper" not in workflow
    assert "runtimes/canary" not in workflow
    assert "runtimes/faster-whisper" not in workflow
    assert "-canary" not in workflow
    assert "-faster-whisper" not in workflow