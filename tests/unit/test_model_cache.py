"""Focused tests for the shared offline Hugging Face snapshot resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from speech_transcriber.errors import ModelLoadError
from speech_transcriber.model_cache import resolve_hf_snapshot


def write_repo(cache: Path, name: str, *, revisions: list[str], main: str | None) -> None:
    repo = cache / "hub" / f"models--{name}"
    for revision in revisions:
        (repo / "snapshots" / revision).mkdir(parents=True, exist_ok=True)
    if main is not None:
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(f"{main}\n", encoding="utf-8")


def reference(name: str) -> str:
    return name.replace("--", "/")


def test_explicit_absolute_local_directory_is_returned_untouched(tmp_path: Path) -> None:
    local = tmp_path / "voxtral"
    local.mkdir()

    assert resolve_hf_snapshot(str(local)) == local
    assert resolve_hf_snapshot(str(local), offline=True) == local


def test_repository_id_uses_cache_via_refs_main(tmp_path: Path) -> None:
    write_repo(
        tmp_path, "mistralai--Voxtral-Mini-4B-Realtime-2602", revisions=["abc123"], main="abc123"
    )

    resolved = resolve_hf_snapshot(
        "mistralai/Voxtral-Mini-4B-Realtime-2602", hf_home=str(tmp_path)
    )

    assert resolved == tmp_path / "hub" / "models--mistralai--Voxtral-Mini-4B-Realtime-2602" / (
        "snapshots/abc123"
    )


def test_refs_main_wins_when_multiple_snapshots_exist(tmp_path: Path) -> None:
    write_repo(
        tmp_path, "mistralai--Voxtral", revisions=["old123", "new456"], main="new456"
    )

    resolved = resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))

    assert resolved == tmp_path / "hub" / "models--mistralai--Voxtral" / "snapshots/new456"


def test_exactly_one_snapshot_without_refs_main_is_deterministic(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["abc123"], main=None)

    resolved = resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))

    assert resolved == tmp_path / "hub" / "models--mistralai--Voxtral" / "snapshots/abc123"


def test_multiple_snapshots_without_refs_main_fail(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["old123", "new456"], main=None)

    with pytest.raises(ModelLoadError, match="refusing to guess"):
        resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))


def test_missing_repository_fails_offline(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path / "cache"))


def test_missing_repository_without_hf_home_fails_offline(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path / "cache"), offline=True)


def test_falls_back_to_reference_when_online_without_cache(tmp_path: Path) -> None:
    assert resolve_hf_snapshot(
        "mistralai/Voxtral",
        hf_home=str(tmp_path / "cache"),
        offline=False,
        fallback_to_reference=True,
    ) == Path("mistralai/Voxtral")


def test_fallback_never_happens_offline(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_hf_snapshot(
            "mistralai/Voxtral",
            hf_home=str(tmp_path / "cache"),
            offline=True,
            fallback_to_reference=True,
        )


def test_refs_main_pointing_at_missing_snapshot_is_ignored(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["abc123"], main="missing999")

    resolved = resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))

    assert resolved == tmp_path / "hub" / "models--mistralai--Voxtral" / "snapshots/abc123"


def test_refs_main_pointing_at_missing_snapshot_fails_with_ambiguity(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["old123", "abc123"], main="missing999")

    with pytest.raises(ModelLoadError, match="refusing to guess"):
        resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))


def test_refs_main_with_no_snapshots_fails_as_cache_miss(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=[], main="abc123")

    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))


def test_empty_refs_main_behaves_like_missing_ref(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["abc123"], main=None)
    repo = tmp_path / "hub" / "models--mistralai--Voxtral"
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("   \n", encoding="utf-8")

    resolved = resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))

    assert resolved == tmp_path / "hub" / "models--mistralai--Voxtral" / "snapshots/abc123"


def test_malformed_refs_main_with_traversal_falls_back_deterministically(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["abc123"], main="../../etc")
    (tmp_path / "hub" / "models--mistralai--Voxtral" / "snapshots" / "extra").mkdir(
        parents=True
    )

    with pytest.raises(ModelLoadError, match="refusing to guess"):
        resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))


def test_malformed_refs_main_with_single_snapshot_still_resolves(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["abc123"], main="not/a/revision")

    resolved = resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))

    assert resolved == tmp_path / "hub" / "models--mistralai--Voxtral" / "snapshots/abc123"


def test_ambiguous_cache_fails_even_if_refs_main_names_one_snapshot(
    tmp_path: Path,
) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["old123", "new456"], main=None)

    with pytest.raises(ModelLoadError, match="refusing to guess"):
        resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))


def test_empty_snapshots_directory_of_existing_repo_fails(tmp_path: Path) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=[], main=None)

    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_hf_snapshot("mistralai/Voxtral", hf_home=str(tmp_path))


def test_environment_hf_home_used_when_keyword_absent(
    tmp_path: Path, monkeypatch: object
) -> None:
    write_repo(tmp_path, "mistralai--Voxtral", revisions=["abc123"], main="abc123")
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # type: ignore[attr-defined]

    resolved = resolve_hf_snapshot("mistralai/Voxtral")

    assert resolved.name == "abc123"