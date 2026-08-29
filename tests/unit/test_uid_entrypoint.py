"""Static checks for arbitrary-UID container identity support."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]

CONTAINERFILES = ("Containerfile", "Containerfile.faster-whisper", "Containerfile.canary")


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
        assert '"/app/.venv/bin/python", "-m", "speech_transcriber"' in container


def test_canary_non_root_smoke_performs_passwd_lookup() -> None:
    container = (ROOT / "Containerfile.canary").read_text(encoding="utf-8")
    assert "pwd.getpwuid(os.getuid())" in container
    assert "USER 1001" in container
    assert "uid-entrypoint" in container


def test_uid_entrypoint_handles_unknown_and_existing_uids() -> None:
    entrypoint = (ROOT / "container/uid-entrypoint.sh").read_text(encoding="utf-8")
    assert "getent passwd" in entrypoint
    assert "getent group" in entrypoint
    assert "NSS_WRAPPER_PASSWD" in entrypoint
    assert "NSS_WRAPPER_GROUP" in entrypoint
    assert "LD_PRELOAD" in entrypoint
    assert "libnss_wrapper.so" in entrypoint
    assert 'exec "$@"' in entrypoint
    assert "cat /etc/passwd" in entrypoint
    assert "cat /etc/group" in entrypoint
    assert "NSS_WRAPPER_TMPDIR" in entrypoint
    assert "HOME:-/cache/home" in entrypoint


def test_uid_entrypoint_uses_unique_temporary_identity_files() -> None:
    entrypoint = (ROOT / "container/uid-entrypoint.sh").read_text(encoding="utf-8")
    assert 'mktemp "$nss_tmpdir/nss-passwd.XXXXXX"' in entrypoint
    assert 'mktemp "$nss_tmpdir/nss-group.XXXXXX"' in entrypoint
    assert 'nss_tmpdir="${NSS_WRAPPER_TMPDIR:-/tmp}"' in entrypoint
    assert "${TMPDIR:-/tmp}" not in entrypoint
    assert "${TMPDIR:-/tmp}/passwd" not in entrypoint
    assert "${TMPDIR:-/tmp}/group" not in entrypoint
    assert "cp /etc/passwd" not in entrypoint
    assert "cp /etc/group" not in entrypoint


def test_ci_smoke_tests_arbitrary_uid() -> None:
    workflow = (ROOT / ".github/workflows/container.yml").read_text(encoding="utf-8")
    assert "--user 12345:0" in workflow
    assert "pwd.getpwuid(os.getuid())" in workflow
    assert "load: true" in workflow
    assert "--tmpfs /cache" in workflow
