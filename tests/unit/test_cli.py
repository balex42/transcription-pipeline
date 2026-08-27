from meeting_transcriber import cli
from meeting_transcriber.config import DEFAULT_PYANNOTE_MODEL, DEFAULT_QWEN_ALIGNER_MODEL


def test_prefetch_whisper_includes_the_forced_aligner(monkeypatch: object) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def prefetch(asr: str, pyannote: str, aligner: str | None) -> None:
        calls.append((asr, pyannote, aligner))

    monkeypatch.setattr(cli, "_prefetch", prefetch)  # type: ignore[attr-defined]

    assert cli.main(["prefetch-models", "--asr", "whisper"]) == 0
    assert calls == [
        ("openai/whisper-large-v3", DEFAULT_PYANNOTE_MODEL, DEFAULT_QWEN_ALIGNER_MODEL)
    ]
