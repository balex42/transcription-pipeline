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


def test_silence_tag_timestamps_count_towards_timestamp_count() -> None:
    """The true tag total includes silence markers, matching generated tokens."""
    metadata = parse_granite_timestamp_words("hallo [T:45] _ [T:120] welt [T:200]")

    assert metadata.timestamp_count == 3
    assert metadata.silence_marker_count == 1


def test_out_of_range_timestamp_tag_fails_instead_of_guessing() -> None:
    """Tags above 999 cannot be modulo-1000 output and indicate corruption."""
    with pytest.raises(ASROutputError, match=r"exceeds the three-digit modulo-1000 range"):
        parse_granite_timestamp_words("hallo [T:1450] welt [T:1620]")


def test_tag_after_silence_never_closes_a_lexical_word() -> None:
    """A `_` timing record does not consume the pending word's later tag."""
    metadata = parse_granite_timestamp_words("hallo _ [T:200] welt [T:300]")

    assert [(word.text, word.end) for word in metadata.words] == [
        ("hallo", 2.0),
        ("welt", 3.0),
    ]


def test_leading_silence_marker_is_ignored() -> None:
    metadata = parse_granite_timestamp_words("_ [T:60] hallo [T:120]")

    assert [(word.text, word.end) for word in metadata.words] == [("hallo", 1.2)]
    assert metadata.silence_marker_count == 1


def test_word_searches_including_multibyte_text_are_preserved() -> None:
    metadata = parse_granite_timestamp_words(
        "Zuckersteuer [T:830] Bundesländer [T:950] 'rum [T:620]"
    )

    assert [word.text for word in metadata.words] == ["Zuckersteuer", "Bundesländer", "'rum"]
    assert all(word.start is None for word in metadata.words)
    assert metadata.timestamp_count == 3


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
    words, metrics = reconcile_segment_end_words(
        segments, {0: [ASRWord(text="Hallo", start=None, end=1.25)]}
    )

    assert [(word.text, word.start, word.end) for word in words] == [("Hallo", None, 181.25)]
    assert metrics["seam_deduplicated_words"] == 0.0
    assert metrics["seam_match_count"] == 0.0
    assert metrics["seam_words_recovered"] == 0.0
    assert metrics["reconciliation_clipped_word_ends"] == 0.0


def test_segment_reconciliation_never_fills_starts_with_segment_starts() -> None:
    segments = [segment(0.0, 180.0, 0), segment(165.0, 345.0, 1)]
    words, _ = reconcile_segment_end_words(
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
    # No lexical seam match (single shared token), so midpoint rules apply.
    assert [(word.text, word.start, word.end) for word in words] == [
        ("eins", None, 170.0),
        ("zwei", None, 175.0),
        ("drei", None, 185.0),
    ]


def segment_words(text: str, *ends: float) -> list[ASRWord]:
    return [ASRWord(text=f"{text}{index}", start=None, end=end) for index, end in enumerate(ends)]


def test_segment_reconciliation_deduplicates_overlap_words() -> None:
    segments = [
        segment(0.0, 180.0, 0),
        segment(165.0, 345.0, 1),
        segment(330.0, 460.0, 2),
    ]
    words, _ = reconcile_segment_end_words(
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


def test_segment_reconciliation_seam_match_without_surviving_duplicates() -> None:
    """Clocks agreeing at a seam matches words but needs no seam action."""
    segments = [segment(0.0, 180.0, 0), segment(165.0, 345.0, 1)]
    words, metrics = reconcile_segment_end_words(
        segments,
        {
            0: [
                ASRWord(text="weit", start=None, end=171.47),
                ASRWord(text="weit", start=None, end=171.66),
                ASRWord(text="aus", start=None, end=171.80),
                ASRWord(text="dem", start=None, end=171.93),
                ASRWord(text="fenster", start=None, end=172.29),
            ],
            1: [
                ASRWord(text="weit", start=None, end=6.47),
                ASRWord(text="weit", start=None, end=6.66),
                ASRWord(text="aus", start=None, end=6.80),
                ASRWord(text="dem", start=None, end=6.93),
                ASRWord(text="fenster", start=None, end=7.29),
                ASRWord(text="gelehnt", start=None, end=7.60),
            ],
        },
    )

    # The b copies rebase onto the a timestamps; midpoint (172.5) already
    # gives every word exactly one survivor, so the seam match only emits
    # telemetry and drops nothing.
    assert [(word.text, word.end) for word in words] == [
        ("weit", 171.47),
        ("weit", 171.66),
        ("aus", 171.80),
        ("dem", 171.93),
        ("fenster", 172.29),
        ("gelehnt", 172.60),
    ]
    assert metrics["seam_matched_words"] == 5.0
    assert metrics["seam_deduplicated_words"] == 0.0
    assert metrics["seam_clock_offset_seconds"] == 0.0


def test_segment_reconciliation_drops_duplicates_when_clocks_disagree() -> None:
    """Off-midpoint duplicates from a later clock keep only the a copy."""
    segments = [segment(0.0, 180.0, 0), segment(165.0, 345.0, 1)]
    words, metrics = reconcile_segment_end_words(
        segments,
        {
            0: [
                ASRWord(text="hat", start=None, end=170.62),
                ASRWord(text="sich", start=None, end=170.77),
                ASRWord(text="da", start=None, end=170.98),
                ASRWord(text="weit", start=None, end=171.25),
                ASRWord(text="weit", start=None, end=171.42),
                ASRWord(text="aus", start=None, end=171.56),
                ASRWord(text="dem", start=None, end=171.70),
                ASRWord(text="fenster", start=None, end=172.06),
                ASRWord(text="gelehnt", start=None, end=172.38),
            ],
            # The same physical words, one second later on seg1's clock, so
            # the b copies of aus/dem/fenster/gelehnt rebase past the
            # midpoint where both sides would otherwise survive.
            1: [
                ASRWord(text="hat", start=None, end=6.62),
                ASRWord(text="sich", start=None, end=6.77),
                ASRWord(text="da", start=None, end=6.98),
                ASRWord(text="weit", start=None, end=7.25),
                ASRWord(text="weit", start=None, end=7.42),
                ASRWord(text="aus", start=None, end=7.56),
                ASRWord(text="dem", start=None, end=7.70),
                ASRWord(text="fenster", start=None, end=8.06),
                ASRWord(text="gelehnt", start=None, end=8.38),
                ASRWord(text="er", start=None, end=8.46),
            ],
        },
    )

    texts = [word.text for word in words]
    for token in ("hat", "aus", "dem", "fenster", "gelehnt", "er"):
        assert texts.count(token) == 1, texts
    assert metrics["seam_match_count"] == 1.0
    assert metrics["seam_deduplicated_words"] == 4.0
    assert metrics["seam_clock_offset_seconds"] == -1.0


def test_segment_reconciliation_protects_earlier_copy_beyond_midpoint() -> None:
    """A matched run midpoint would lose on both sides keeps one survivor."""
    segments = [segment(0.0, 180.0, 0), segment(165.0, 345.0, 1)]
    words, metrics = reconcile_segment_end_words(
        segments,
        {
            0: [
                ASRWord(text="zwei", start=None, end=173.0),
                ASRWord(text="drei", start=None, end=173.5),
            ],
            # Seg1's clock runs two seconds ahead, so its copies rebase to
            # 167/167.5, below the midpoint where seg1 no longer owns them,
            # while the a copies sit beyond the midpoint where seg0 no
            # longer owns them: plain midpoint would lose both words.
            1: [
                ASRWord(text="zwei", start=None, end=2.0),
                ASRWord(text="drei", start=None, end=2.5),
                ASRWord(text="vier", start=None, end=9.0),
            ],
        },
    )

    texts = [word.text for word in words]
    assert texts.count("zwei") == 1
    assert texts.count("drei") == 1
    assert metrics["seam_words_recovered"] == 2.0


def test_segment_reconciliation_requires_multi_word_run_for_dedup() -> None:
    """A single shared token is not enough to deduplicate a seam."""
    segments = [segment(0.0, 180.0, 0), segment(165.0, 345.0, 1)]
    words, metrics = reconcile_segment_end_words(
        segments,
        {
            0: [
                ASRWord(text="eins", start=None, end=171.0),
                ASRWord(text="und", start=None, end=173.0),
            ],
            1: [
                ASRWord(text="und", start=None, end=6.0),
                ASRWord(text="zwei", start=None, end=9.0),
            ],
        },
    )

    # Midpoint at 172.5 decides both single-token copies without a seam match:
    # seg0's "und" (173.0) lands beyond it and seg1's copy rebases below it.
    assert [(word.text, word.end) for word in words] == [
        ("eins", 171.0),
        ("zwei", 174.0),
    ]
    assert metrics["seam_deduplicated_words"] == 0.0


def test_segment_reconciliation_counts_clipped_word_ends() -> None:
    segments = [segment(0.0, 180.0, 0)]
    words, metrics = reconcile_segment_end_words(
        segments,
        {
            0: [
                ASRWord(text="wort", start=None, end=99.0),
                ASRWord(text="x", start=None, end=500.0),
            ]
        },
    )

    assert [(word.text, word.end) for word in words] == [("wort", 99.0), ("x", 180.0)]
    assert metrics["reconciliation_clipped_word_ends"] == 1.0


def test_segment_reconciliation_first_and_last_segments_have_open_boundaries() -> None:
    segments = [segment(0.0, 100.0, 0)]
    words, _ = reconcile_segment_end_words(
        segments, {0: [ASRWord(text="wort", start=None, end=99.0)]}
    )
    assert [(word.text, word.end) for word in words] == [("wort", 99.0)]


def test_segment_reconciliation_global_ends_stay_monotonic() -> None:
    segments = [segment(0.0, 50.0, 0), segment(35.0, 85.0, 1)]
    words, _ = reconcile_segment_end_words(
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
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, object]], **kwargs: object
    ) -> str:
        self.calls.append((conversation, kwargs))
        assert kwargs.get("add_generation_prompt") is True
        assert kwargs.get("tokenize") is False
        content = conversation[-1]["content"]
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


class FakeSegmenter:
    """Stand-in that returns prebuilt segments without touching disk."""

    def __init__(self, segments: list[AudioSegment]) -> None:
        self.segments = segments

    def segment(self, _: NormalizedAudio) -> list[AudioSegment]:
        return self.segments


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
    assert call["audio"] == [segment.audio]
    assert "sampling_rate" not in call
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["num_beams"] == 1
    assert model.calls[0]["max_new_tokens"] == transcriber.backend_configuration["max_new_tokens"]


def test_generate_sends_official_system_and_timestamp_user_prompts() -> None:
    import numpy as np

    transcriber, processor, model = loaded_transcriber()
    segment = AudioSegment(
        index=0, start=0.0, end=1.0, audio=np.zeros(16000, dtype="float32")
    )

    transcriber._generate_segment(processor, model, segment)

    conversation, _ = processor.tokenizer.calls[0]
    assert [entry["role"] for entry in conversation] == ["system", "user"]
    assert conversation[0]["content"] == (
        "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\n"
        "You are Granite, developed by IBM. You are a helpful AI assistant"
    )
    user_content = conversation[1]["content"]
    assert isinstance(user_content, list)
    prompt_text = user_content[0]["text"]
    assert "Timestamps: Transcribe the speech" in str(prompt_text)


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
    assert "peft" not in transcriber.runtime_provenance.components


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


def test_transcribe_reports_full_timestamp_tag_and_seam_diagnostics(tmp_path: Path) -> None:
    import numpy as np

    transcriber, _, _ = loaded_transcriber()
    transcriber._segmenter = FakeSegmenter(
        [AudioSegment(index=0, start=0.0, end=180.0, audio=np.zeros(16, dtype="float32"))]
    )
    transcriber._processor = FakeProcessor("wort [T:45] _ [T:120] welt [T:82]")
    transcriber._model = FakeGenerateModel()

    words = transcriber.transcribe(audio(tmp_path, 180.0))

    assert len(words) == 2
    assert transcriber.backend_metrics["timestamp_tags_decoded"] == 3.0
    assert transcriber.backend_metrics["silence_markers_ignored"] == 1.0
    assert transcriber.backend_metrics["seam_deduplicated_words"] == 0.0
    assert transcriber.backend_metrics["reconciliation_clipped_word_ends"] == 0.0


def test_release_drops_model_and_processor() -> None:
    transcriber, _, _ = loaded_transcriber()
    transcriber.release()
    assert transcriber._model is None
    assert transcriber._processor is None