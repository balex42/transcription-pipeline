"""Unit tests for the Granite Speech 4.1 Plus backend without loading the real model."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.models import ASRWord, AudioMetadata, AudioSegment, NormalizedAudio
from speech_transcriber.transcription.granite import (
    GraniteTranscriber,
    parse_granite_timestamp_words,
)
from speech_transcriber.transcription.segments import reconcile_segment_end_words


def audio(tmp_path: Path, seconds: float = 1.0) -> NormalizedAudio:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * int(16_000 * seconds))
    return NormalizedAudio(path, AudioMetadata(path.name, seconds))


def fake_repository_cache(tmp_path: Path) -> Path:
    """Build a prefetched single-revision HF cache for the Granite model."""
    repository = "models--ibm-granite--granite-speech-4.1-2b-plus"
    snapshot = tmp_path / "hub" / repository / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    refs = tmp_path / "hub" / repository / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc123\n", encoding="utf-8")
    return snapshot


def segment(start: float, end: float, index: int = 0) -> AudioSegment:
    return AudioSegment(index=index, start=start, end=end, audio=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Timestamp parser
# ---------------------------------------------------------------------------


def test_normal_words_get_end_only_timestamps() -> None:
    metadata = parse_granite_timestamp_words("hallo [T:45] welt [T:82]")

    assert [(word.text, word.start, word.end) for word in metadata.words] == [
        ("hallo", None, 0.45),
        ("welt", None, 0.82),
    ]
    assert metadata.rollover_count == 0
    assert metadata.timestamp_count == 2


def test_rollover_wraps_to_the_next_ten_second_window() -> None:
    metadata = parse_granite_timestamp_words("eins [T:950] zwei [T:20] drei [T:130]")

    assert [(word.text, word.end) for word in metadata.words] == [
        ("eins", 9.5),
        ("zwei", 10.2),
        ("drei", 11.3),
    ]
    assert metadata.rollover_count == 1


def test_multiple_rollovers_across_tens_of_seconds() -> None:
    text = (
        "eins [T:930] zwei [T:80] drei [T:970] vier [T:10] fuenf [T:340] sech [T:5]"
    )
    metadata = parse_granite_timestamp_words(text)

    assert [(word.text, word.end) for word in metadata.words] == [
        ("eins", 9.3),
        ("zwei", 10.8),
        ("drei", 19.7),
        ("vier", 20.1),
        ("fuenf", 23.4),
        ("sech", 30.05),
    ]
    assert metadata.rollover_count == 3


def test_silence_markers_never_become_words_or_close_words() -> None:
    # A silence marker between two words carries its own [T:N], which is a
    # timing record for the pause, not a transcript word.
    metadata = parse_granite_timestamp_words("hallo [T:45] _ [T:120] welt [T:200]")

    assert [(word.text, word.end) for word in metadata.words] == [
        ("hallo", 0.45),
        ("welt", 2.0),
    ]
    assert metadata.silence_marker_count == 1


def test_leading_silence_marker_is_ignored() -> None:
    metadata = parse_granite_timestamp_words("_ [T:60] hallo [T:120]")

    assert [(word.text, word.end) for word in metadata.words] == [("hallo", 1.2)]
    assert metadata.silence_marker_count == 1


def test_word_searches_including_multibyte_text_are_preserved() -> None:
    metadata = parse_granite_timestamp_words(
        "Zuckersteuer [T:830] Bundesländer [T:1450] 'rum [T:1620]"
    )

    assert [word.text for word in metadata.words] == ["Zuckersteuer", "Bundesländer", "'rum"]
    assert all(word.start is None for word in metadata.words)


@pytest.mark.parametrize(
    "text",
    [
        "hallo [T:abc]",
        "hallo [T:]",
        "hallo [T:12x]",
    ],
)
def test_non_numeric_timestamps_are_unmatched_and_leave_dangling_words(text: str) -> None:
    # A malformed tag is just lexical text, so its word never closes.
    with pytest.raises(ASROutputError, match="no closing timestamp tag"):
        parse_granite_timestamp_words(text)


def test_text_without_any_timestamp_fails() -> None:
    with pytest.raises(ASROutputError, match="no closing timestamp tag"):
        parse_granite_timestamp_words("naked words without timestamps")


def test_trailing_word_without_timestamp_fails() -> None:
    with pytest.raises(ASROutputError, match="has no closing timestamp tag"):
        parse_granite_timestamp_words("hallo [T:45] welt")


def test_timestamp_without_preceding_word_is_a_pause_record() -> None:
    """A leading tag carries pause timing only; no fabricated word is emitted."""
    metadata = parse_granite_timestamp_words("[T:45]")

    assert metadata.words == []
    assert metadata.silence_marker_count == 1


def test_backwards_tag_sequence_after_completed_word_fails() -> None:
    # A second tag directly after a closed word (no pending lexical term)
    # cannot rewind time: unwrapping only ever moves time forward.
    with pytest.raises(ASROutputError, match="no closing timestamp tag"):
        parse_granite_timestamp_words("eins [T:950] zwei [T:20] drei [T:10] extra")


def test_rollover_unwrap_is_monotonic_by_construction() -> None:
    """Any well-formed tag sequence yields non-decreasing absolute ends."""
    # [T:10] after [T:20] does not roll back: 10 is not below 20 % 1000, so
    # the parser reads a new wrap (20.10s) rather than rewinding to 11.10s.
    metadata = parse_granite_timestamp_words("eins [T:950] zwei [T:20] drei [T:10]")
    ends = [word.end for word in metadata.words]
    assert ends == sorted(ends) == [9.5, 10.2, 20.1]


def test_empty_text_yields_no_words() -> None:
    metadata = parse_granite_timestamp_words("")
    assert metadata.words == []
    assert metadata.timestamp_count == 0


def test_whitespace_around_tags_is_tolerated() -> None:
    metadata = parse_granite_timestamp_words("  hallo   [T:45]   welt  [T:82] ")

    assert [(word.text, word.end) for word in metadata.words] == [
        ("hallo", 0.45),
        ("welt", 0.82),
    ]


def test_punctuation_is_not_required_by_the_parser() -> None:
    metadata = parse_granite_timestamp_words("und [T:30] aber [T:60]")
    assert [word.text for word in metadata.words] == ["und", "aber"]


# ---------------------------------------------------------------------------
# End-only segment reconciliation
# ---------------------------------------------------------------------------


def test_segment_reconciliation_rebases_ends_and_keeps_start_none() -> None:
    segments = [segment(180.0, 360.0)]
    words = reconcile_segment_end_words(
        segments, {0: [ASRWord(text="Hallo", start=None, end=1.25)]}
    )

    assert [(word.text, word.start, word.end) for word in words] == [("Hallo", None, 181.25)]


def test_segment_reconciliation_never_fills_starts_with_segment_starts() -> None:
    segments = [segment(0.0, 180.0), segment(165.0, 345.0, index=1)]
    words = reconcile_segment_end_words(
        segments,
        {
            0: [
                ASRWord(text="eins", start=None, end=170.0),
                ASRWord(text="zwei", start=None, end=175.0),
            ],
            1: [
                ASRWord(text="zwei", start=None, end=10.0),
                ASRWord(text="drei", start=None, end=20.0),
            ],
        },
    )

    # Overlap midpoint is (165 + 180) / 2 = 172.5; ownership is by rebased end.
    assert [(word.text, word.start, word.end) for word in words] == [
        ("eins", None, 170.0),
        ("zwei", None, 175.0),
        ("drei", None, 185.0),
    ]


def segment_words(text: str, *ends: float) -> list[ASRWord]:
    return [ASRWord(text=f"{text}{index}", start=None, end=end) for index, end in enumerate(ends)]


def test_segment_reconciliation_deduplicates_overlap_words() -> None:
    segments = [
        segment(0.0, 180.0),
        segment(165.0, 345.0, index=1),
        segment(330.0, 460.0, index=2),
    ]
    words = reconcile_segment_end_words(
        segments,
        {
            0: segment_words("s0", 171.0, 179.0),
            1: segment_words("s1", 6.0, 30.0, 171.0),
            2: segment_words("s2", 1.5, 10.0),
        },
    )

    # Midpoint ownership: seg0 keeps ends < 172.5, seg1 keeps [172.5, 337.5),
    # seg2 keeps >= 337.5. Each lexical word survives exactly once.
    assert [(word.text, word.end) for word in words] == [
        ("s00", 171.0),
        ("s11", 195.0),
        ("s12", 336.0),
        ("s21", 340.0),
    ]


def test_segment_reconciliation_first_and_last_segments_have_open_boundaries() -> None:
    segments = [segment(0.0, 100.0)]
    words = reconcile_segment_end_words(segments, {0: [ASRWord(text="wort", start=None, end=99.0)]})
    assert [(word.text, word.end) for word in words] == [("wort", 99.0)]


def test_segment_reconciliation_global_ends_stay_monotonic() -> None:
    segments = [segment(0.0, 50.0), segment(35.0, 85.0, index=1)]
    words = reconcile_segment_end_words(
        segments,
        {
            0: [ASRWord(text="a", start=None, end=10.0), ASRWord(text="b", start=None, end=42.0)],
            1: [ASRWord(text="b", start=None, end=52.0), ASRWord(text="c", start=None, end=60.0)],
        },
    )
    ends = [word.end for word in words]
    assert ends == sorted(ends)


# ---------------------------------------------------------------------------
# Offline snapshot resolution (shared resolver paths)
# ---------------------------------------------------------------------------


def test_repository_id_resolves_through_local_hf_cache_refs_main(
    tmp_path: Path, monkeypatch: object
) -> None:
    from speech_transcriber.transcription.granite import resolve_granite_model_path

    snapshot = fake_repository_cache(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # type: ignore[attr-defined]

    assert resolve_granite_model_path("ibm-granite/granite-speech-4.1-2b-plus") == str(snapshot)


def test_repository_id_uses_single_snapshot_without_refs_main(
    tmp_path: Path, monkeypatch: object
) -> None:
    from speech_transcriber.transcription.granite import resolve_granite_model_path

    snapshot = (
        tmp_path / "hub" / "models--ibm-granite--granite-speech-4.1-2b-plus"
        / "snapshots" / "xyz789"
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # type: ignore[attr-defined]

    assert resolve_granite_model_path("ibm-granite/granite-speech-4.1-2b-plus") == str(snapshot)


def test_repository_id_fails_without_online_fallback(
    tmp_path: Path, monkeypatch: object
) -> None:
    from speech_transcriber.transcription.granite import resolve_granite_model_path

    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))  # type: ignore[attr-defined]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # type: ignore[attr-defined]

    with pytest.raises(ModelLoadError, match="not present in the offline model cache"):
        resolve_granite_model_path("ibm-granite/granite-speech-4.1-2b-plus")


def test_ambiguous_snapshots_fail_deterministically(tmp_path: Path, monkeypatch: object) -> None:
    from speech_transcriber.transcription.granite import resolve_granite_model_path

    snapshots = (
        tmp_path / "hub" / "models--ibm-granite--granite-speech-4.1-2b-plus" / "snapshots"
    )
    (snapshots / "old123").mkdir(parents=True)
    (snapshots / "new456").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # type: ignore[attr-defined]

    with pytest.raises(ModelLoadError, match="refusing to guess"):
        resolve_granite_model_path("ibm-granite/granite-speech-4.1-2b-plus")


def test_explicit_local_directory_is_used_directly(tmp_path: Path) -> None:
    from speech_transcriber.transcription.granite import resolve_granite_model_path

    local = tmp_path / "granite"
    local.mkdir()

    assert resolve_granite_model_path(str(local)) == str(local)


def test_missing_absolute_directory_fails_clearly(tmp_path: Path) -> None:
    from speech_transcriber.transcription.granite import resolve_granite_model_path

    with pytest.raises(ModelLoadError, match="does not exist"):
        resolve_granite_model_path(str(tmp_path / "missing"))


# ---------------------------------------------------------------------------
# Transcriber load / transcribe behavior against fake Transformers classes
# ---------------------------------------------------------------------------


class FakeShapedInputs(dict[str, object]):
    """Stand-in exposing an input_ids shape without importing PyTorch."""

    def __init__(self) -> None:
        super().__init__(input_ids=_ShapedShape())

    def to(self, device: str, dtype: object) -> FakeShapedInputs:
        return self


class _ShapedShape:
    @property
    def shape(self) -> tuple[int, ...]:
        return (1, 5)

    def __len__(self) -> int:
        return 5


class FakeTokenizer:
    def apply_chat_template(
        self, conversation: list[dict[str, object]], **kwargs: object
    ) -> str:
        assert kwargs.get("add_generation_prompt") is True
        assert kwargs.get("tokenize") is False
        content = conversation[0]["content"]
        assert isinstance(content, list)
        text = content[-1]["text"]
        assert isinstance(text, str)
        return f"<prompt><|audio|>{text}<end>"


class FakeProcessor:
    def __init__(self, transcript: str = "hallo [T:45] welt [T:82]") -> None:
        self.tokenizer = FakeTokenizer()
        self.transcript = transcript
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, audio: object, **kwargs: object) -> FakeShapedInputs:
        self.calls.append({"text": text, "audio": audio, **kwargs})
        return FakeShapedInputs()

    def batch_decode(self, token_ids: object, **kwargs: object) -> list[str]:
        assert kwargs == {"skip_special_tokens": True}
        return [self.transcript]


class FakeGenerateModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def to(self, _: str) -> FakeGenerateModel:
        return self

    def eval(self) -> FakeGenerateModel:
        return self

    def generate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return _GraniteOutput()


class _GraniteOutput:
    """Generation output shaped like a batch tensor of 35 prompt+new tokens."""

    class _Seq:
        def __getitem__(self, key: object) -> Any:
            return self

        @property
        def shape(self) -> tuple[int, ...]:
            return (1, 35)

        def numel(self) -> int:
            return 35

    sequences = _Seq()


def loaded_transcriber(
    model_reference: str = "/models/granite",
    device: str = "cpu",
    segment_duration: float = 180.0,
    segment_overlap: float = 15.0,
) -> tuple[GraniteTranscriber, FakeProcessor, FakeGenerateModel]:
    """Return a transcriber pre-loaded with fake Transformers stand-ins."""
    transcriber = GraniteTranscriber(model_reference, device, segment_duration, segment_overlap)
    processor = FakeProcessor()
    model = FakeGenerateModel()
    transcriber._processor = processor  # type: ignore[attr-defined]
    transcriber._model = model  # type: ignore[attr-defined]
    return transcriber, processor, model


def test_generate_uses_official_recipe_with_deterministic_timestamp_prompt() -> None:
    import numpy as np

    transcriber, processor, model = loaded_transcriber()
    segment = AudioSegment(
        index=0, start=0.0, end=1.0, audio=np.zeros(16000, dtype="float32")
    )

    text = transcriber._generate_segment(processor, model, segment)

    assert text == "hallo [T:45] welt [T:82]"
    call = processor.calls[0]
    assert "Timestamps: Transcribe the speech" in str(call["text"])
    assert call["sampling_rate"] == 16000
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == transcriber.backend_configuration["max_new_tokens"]


def test_snapshot_resolution_feeds_one_local_path_to_processor_and_model(
    tmp_path: Path, monkeypatch: object
) -> None:
    snapshot = fake_repository_cache(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # type: ignore[attr-defined]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # type: ignore[attr-defined]
    calls: list[tuple[str, str, dict[str, object]]] = []

    class RecordingProcessor(FakeProcessor):
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeProcessor:
            calls.append(("processor", path, kwargs))
            return cls()

    class RecordingModel(FakeGenerateModel):
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> RecordingModel:
            calls.append(("model", path, kwargs))
            return cls()

    import speech_transcriber.transcription.granite as module

    monkeypatch.setattr(  # type: ignore[attr-defined]
        module,
        "_transformers_factories",
        lambda: (RecordingProcessor, RecordingModel),
    )
    transcriber = GraniteTranscriber("ibm-granite/granite-speech-4.1-2b-plus", "cpu")
    transcriber.load()

    for _, path, kwargs in calls:
        assert path == str(snapshot)
        assert kwargs.get("trust_remote_code") is False
        assert kwargs.get("local_files_only") is True
    assert transcriber.backend_models["model_snapshot"] == "abc123"
    assert transcriber.runtime_provenance.components["peft"] != "unknown"


def test_load_wraps_transformers_failures_as_model_load_errors(
    tmp_path: Path, monkeypatch: object
) -> None:
    class ExplodingProcessor(FakeProcessor):
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            raise RuntimeError("snapshot incomplete")

    import speech_transcriber.transcription.granite as module

    monkeypatch.setattr(  # type: ignore[attr-defined]
        module,
        "_transformers_factories",
        lambda: (ExplodingProcessor, FakeGenerateModel),
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # type: ignore[attr-defined]
    transcriber = GraniteTranscriber("/models/granite", "cpu")

    with pytest.raises(ModelLoadError, match="could not load Granite model"):
        transcriber._load()


def test_release_drops_model_and_processor() -> None:
    transcriber, _, _ = loaded_transcriber()
    transcriber.release()
    assert transcriber._model is None
    assert transcriber._processor is None