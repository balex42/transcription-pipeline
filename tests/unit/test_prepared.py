import json
from pathlib import Path

import pytest

from speech_transcriber.models import AudioMetadata, DiarizationSegment, NormalizedAudio
from speech_transcriber.pipeline import PreparedRecording
from speech_transcriber.prepared import load_prepared_recording, write_prepared_recording


def _write_bundle(tmp_path: Path) -> Path:
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
        diarization_model="pyannote/test",
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


def test_load_rejects_missing_prepared_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prepared manifest is missing"):
        load_prepared_recording(tmp_path)


def test_load_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    directory = _write_bundle(tmp_path)
    manifest_path = directory / "prepared.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported prepared artifact schema version"):
        load_prepared_recording(directory)


@pytest.mark.parametrize("missing", ["normalized.wav", "diarization.json"])
def test_load_rejects_missing_required_files(tmp_path: Path, missing: str) -> None:
    directory = _write_bundle(tmp_path)
    (directory / missing).unlink()

    with pytest.raises(ValueError, match="missing"):
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
