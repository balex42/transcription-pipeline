from speech_transcriber import cli
from speech_transcriber.config import DEFAULT_PYANNOTE_MODEL, DEFAULT_QWEN_ALIGNER_MODEL


def test_prefetch_qwen_includes_the_forced_aligner(monkeypatch: object) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def prefetch(asr: str, pyannote: str, aligner: str | None) -> None:
        calls.append((asr, pyannote, aligner))

    monkeypatch.setattr(cli, "_prefetch", prefetch)  # type: ignore[attr-defined]

    assert cli.main(["prefetch-models", "--asr", "qwen"]) == 0
    assert calls == [
        ("Qwen/Qwen3-ASR-1.7B-hf", DEFAULT_PYANNOTE_MODEL, DEFAULT_QWEN_ALIGNER_MODEL)
    ]


def test_compare_defaults_to_all_production_backends() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["compare", "input.wav", "--output", "output"])
    assert args.models == "parakeet,qwen,nemotron,voxtral"
