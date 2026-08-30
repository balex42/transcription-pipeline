import json
from pathlib import Path

import pytest

from speech_transcriber.models import AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.prepared import (
    PreparedRecording,
    load_prepared_recording,
    sha256_file,
    write_prepared_recording,
)


def _write_bundle(tmp_path: Path, diarization_model: str = "pyannote/test") -> Path:
    normalized = tmp_path / "source.wav"
    normalized.write_bytes(b"normalized audio")
    prepared = PreparedRecording(
        audio=NormalizedAudio(
            normalized,
            AudioMetadata(
                source="/private/input/meeting.m4a",
                duration_seconds=12.5,
            ),
        ),
        diarization=[DiarizationSegment("SPEAKER_00", 0.0, 12.5)],
        work_directory=tmp_path,
        normalized_audio_sha256=sha256_file(normalized),
        diarization_model=diarization_model,
        language="de-DE",
    )
    destination = tmp_path / "prepared"
    write_prepared_recording(prepared, destination)
    return destination


def test_prepared_recording_round_trip_uses_relative_paths(tmp_path: Path) -> None:
    directory = _write_bundle(tmp_path)

    loaded = load_prepared_recording(directory)

    assert loaded.audio.metadata == AudioMetadata("meeting.m4a", 12.5)
    assert loaded.audio.path == directory / "normalized.wav"
    assert loaded.diarization == [DiarizationSegment("SPEAKER_00", 0.0, 12.5)]
    assert loaded.diarization_model == "pyannote/test"
    assert loaded.language == "de-DE"
    assert not loaded.cleanup_enabled
    manifest = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    assert manifest["audio"]["file"] == "normalized.wav"
    assert len(manifest["audio"]["sha256"]) == 64
    assert loaded.normalized_audio_sha256 == manifest["audio"]["sha256"]


@pytest.mark.parametrize(
    "model",
    ["pyannote/speaker-diarization-community-1", "/models/pyannote-community-1"],
)
def test_pyannote_model_provenance_round_trips_unchanged(
    tmp_path: Path, model: str
) -> None:
    directory = _write_bundle(tmp_path, diarization_model=model)

    loaded = load_prepared_recording(directory)

    assert loaded.diarization_model == model
    manifest = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    assert manifest["diarization"]["model"] == model


def test_load_rejects_missing_prepared_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prepared manifest is missing"):
        load_prepared_recording(tmp_path)


def test_load_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    directory = _write_bundle(tmp_path)
    manifest_path = directory / "prepared.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported prepared artifact schema version"):
        load_prepared_recording(directory)


@pytest.mark.parametrize("missing", ["normalized.wav", "diarization.json"])
def test_load_rejects_missing_required_files(tmp_path: Path, missing: str) -> None:
    directory = _write_bundle(tmp_path)
    (directory / missing).unlink()

    with pytest.raises(ValueError, match="missing"):
        load_prepared_recording(directory)


def test_load_rejects_invalid_audio_digest(tmp_path: Path) -> None:
    directory = _write_bundle(tmp_path)
    manifest_path = directory / "prepared.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audio"]["sha256"] = "not-a-digest"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_prepared_recording(directory)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("audio", "file", "/tmp/normalized.wav"),
        ("audio", "file", "../normalized.wav"),
        ("diarization", "file", "/tmp/diarization.json"),
        ("diarization", "file", "nested/diarization.json"),
        ("audio", "source", "/private/input/meeting.m4a"),
        ("audio", "source", "../meeting.m4a"),
    ],
)
def test_load_rejects_nonportable_artifact_file_references(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    directory = _write_bundle(tmp_path)
    manifest_path = directory / "prepared.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[section][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative filename"):
        load_prepared_recording(directory)


def test_prepared_recording_requires_a_language() -> None:
    with pytest.raises(TypeError):
        PreparedRecording(  # type: ignore[call-arg]
            audio=NormalizedAudio(Path("audio.wav"), AudioMetadata("audio.wav", 1.0)),
            diarization=[],
            work_directory=Path("work"),
            normalized_audio_sha256="0" * 64,
            diarization_model="pyannote/test",
        )


def test_load_rejects_modified_normalized_audio(tmp_path: Path) -> None:
    directory = _write_bundle(tmp_path)
    (directory / "normalized.wav").write_bytes(b"different normalized audio")

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        load_prepared_recording(directory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("duration_seconds", 0, "duration_seconds"),
        ("sample_rate", 0, "sample_rate"),
        ("sample_rate", 8_000, "16 kHz mono 16-bit PCM WAV"),
    ],
)
def test_load_rejects_invalid_audio_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    directory = _write_bundle(tmp_path)
    manifest_path = directory / "prepared.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audio"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_prepared_recording(directory)


@pytest.mark.parametrize(
    "record",
    [
        {"speaker": "SPEAKER_00", "start": -0.1, "end": 1.0},
        {"speaker": "SPEAKER_00", "start": 2.0, "end": 1.0},
        {"speaker": "SPEAKER_00", "start": "zero", "end": 1.0},
        {"start": 0.0, "end": 1.0},
    ],
)
def test_load_rejects_invalid_diarization_records(tmp_path: Path, record: object) -> None:
    directory = _write_bundle(tmp_path)
    (directory / "diarization.json").write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError, match="diarization"):
        load_prepared_recording(directory)
