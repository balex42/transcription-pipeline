"""Static checks for arbitrary-UID container identity support."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]

CONTAINERFILES = (
    "Containerfile.transformers",
    "Containerfile.nemo",
    "Containerfile.ctranslate2",
)


def test_shared_uid_entrypoint_exists_and_is_executable() -> None:
    entrypoint = ROOT / "container/uid-entrypoint.sh"
    assert entrypoint.is_file()
    assert entrypoint.stat().st_mode & 0o111


def test_all_runtime_images_use_the_shared_uid_entrypoint() -> None:
    for name in CONTAINERFILES:
        container = (ROOT / name).read_text(encoding="utf-8")
        assert "COPY container/uid-entrypoint.sh /usr/local/bin/uid-entrypoint" in container
        entrypoint_line = (
            'ENTRYPOINT ["/usr/local/bin/uid-entrypoint", '
            '"/app/.venv/bin/python", "-m", "speech_transcriber"]'
        )
        assert entrypoint_line in container
        assert 'CMD ["--help"]' in container


def test_all_runtime_images_install_libnss_wrapper() -> None:
    for name in CONTAINERFILES:
        container = (ROOT / name).read_text(encoding="utf-8")
        assert "libnss-wrapper" in container


def test_no_image_creates_a_static_user() -> None:
    for name in CONTAINERFILES:
        container = (ROOT / name).read_text(encoding="utf-8")
        assert "useradd" not in container
        assert "adduser" not in container


def test_home_remains_cache_home_in_all_images() -> None:
    for name in CONTAINERFILES:
        container = (ROOT / name).read_text(encoding="utf-8")
        assert "HOME=/cache/home" in container


def test_entrypoint_preserves_speech_transcriber_invocation() -> None:
    for name in CONTAINERFILES:
        container = (ROOT / name).read_text(encoding="utf-8")
        assert (
            'ENTRYPOINT ["/usr/local/bin/uid-entrypoint", '
            '"/app/.venv/bin/python", "-m", "speech_transcriber"]' in container
        )


def test_nemo_image_smoke_performs_passwd_lookup_for_all_ne_mo_adapters() -> None:
    container = (ROOT / "Containerfile.nemo").read_text(encoding="utf-8")

    smoke = container.split("USER 1001", 1)[1]
    assert "pwd.getpwuid(os.getuid())" in smoke
    for backend in ("parakeet", "primeline", "canary"):
        assert f"speech_transcriber.transcription.{backend}" in smoke
    assert "USER 1001" in container