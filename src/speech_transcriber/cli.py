"""Worker CLI for the prepare, recognize, finalize, and prefetch stages.

Argo Workflows composes these worker commands in production; each command runs
inside its runtime image against filesystem artifacts and never orchestrates
other stages.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from speech_transcriber.asr_artifact import (
    load_asr_recognition,
    write_asr_recognition,
    write_asr_result_files,
)
from speech_transcriber.config import (
    ASR_BACKENDS,
    BACKEND_RUNTIMES,
    DEFAULT_PYANNOTE_MODEL,
    DEFAULT_QWEN_ALIGNER_MODEL,
    FASTER_WHISPER_COMPUTE_TYPES,
    PipelineConfig,
)
from speech_transcriber.errors import TranscriberError
from speech_transcriber.models import ASRRecognitionResult
from speech_transcriber.prepared import PreparedRecording, load_prepared_recording

if TYPE_CHECKING:
    from speech_transcriber.recognition import MemoryMetrics

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

# Complete backend list for parametrized coverage in tests.
BACKEND_TEST_IDS = ASR_BACKENDS


def subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return the parser's worker subparsers by name (test seam for option surface)."""
    for action in parser._actions:  # noqa: SLF001
        if action.dest == "command":
            choices = getattr(action, "choices", None)
            assert isinstance(choices, dict)
            return {name: value for name, value in choices.items()}
    raise AssertionError("parser has no command subparsers")


def build_parser() -> argparse.ArgumentParser:
    """Build the public worker CLI parser."""
    parser = argparse.ArgumentParser(prog="speech-transcriber")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="normalize and diarize a recording into the prepared artifact"
    )
    _add_prepare_options(prepare)

    recognize = commands.add_parser(
        "recognize", help="run one ASR backend over a prepared recording"
    )
    _add_recognize_options(recognize)

    finalize = commands.add_parser(
        "finalize", help="build a speaker-attributed transcript from prepared and ASR artifacts"
    )
    _add_finalize_options(finalize)

    prefetch = commands.add_parser(
        "prefetch", help="download models into the configured Hugging Face cache"
    )
    prefetch.add_argument("--backend", choices=ASR_BACKENDS, required=True)
    prefetch.add_argument("--model", help="Hugging Face model ID or local model directory")
    prefetch.add_argument("--qwen-aligner-model")
    prefetch.add_argument("--pyannote-model")
    prefetch.add_argument("--log-level", choices=_LOG_LEVELS)
    return parser


def _add_prepare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--pyannote-model", help="pyannote model ID or local model directory")
    parser.add_argument("--language", help="recording language locale (default: de-DE)")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--log-level", choices=_LOG_LEVELS)


def _add_recognize_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backend", choices=ASR_BACKENDS, required=True)
    parser.add_argument("--model", help="Hugging Face model ID or local model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--qwen-aligner-model", help="Qwen forced-aligner model ID or local path"
    )
    parser.add_argument("--parakeet-segment-duration", type=float)
    parser.add_argument("--parakeet-segment-overlap", type=float)
    parser.add_argument("--qwen-segment-duration", type=float)
    parser.add_argument("--qwen-segment-overlap", type=float)
    parser.add_argument("--nemotron-num-lookahead-tokens", type=int)
    parser.add_argument(
        "--voxtral-delay-ms", type=int, help="Voxtral streaming delay in ms (default: 2400)"
    )
    parser.add_argument(
        "--voxtral-timestamp-offset-tokens",
        type=int,
        help="Voxtral marker timestamp offset in tokens (default: 4)",
    )
    parser.add_argument(
        "--faster-whisper-compute-type",
        choices=FASTER_WHISPER_COMPUTE_TYPES,
        help="faster-whisper CTranslate2 compute type (default: float16)",
    )
    parser.add_argument(
        "--canary-chunk-duration",
        type=float,
        help="Canary sequential chunk duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--language", help="ASR language locale (default: prepared artifact language)"
    )
    parser.add_argument("--log-level", choices=_LOG_LEVELS)


def _add_finalize_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--backend", choices=ASR_BACKENDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--language", help="transcript language locale (default: prepared artifact language)"
    )
    parser.add_argument("--log-level", choices=_LOG_LEVELS)


def main(argv: list[str] | None = None) -> int:
    """Run one worker stage and return a shell-compatible status code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level or "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        if args.command == "prefetch":
            _prefetch(_prefetch_config(args))
            return 0
        if args.command == "finalize":
            return _finalize(args)
        config = _config_from_args(args)
        if args.command == "prepare":
            return _prepare(config)
        recognition = _recognize(config)
        write_asr_recognition(recognition, config.output_directory)
        return 0
    except (TranscriberError, OSError, RuntimeError, ValueError) as error:
        logging.getLogger(__name__).error("worker command failed: %s", error)
        return 1


def _prepare(config: PipelineConfig) -> int:
    """Normalize and diarize once, then write the versioned prepared artifact."""
    from speech_transcriber.preparation import PreparationRunner
    from speech_transcriber.prepared import write_prepared_recording

    runner = PreparationRunner.create_default(config)
    prepared = runner.prepare()
    try:
        write_prepared_recording(prepared, config.output_directory)
    finally:
        runner.cleanup(prepared)
    return 0


def _recognize(config: PipelineConfig) -> ASRRecognitionResult:
    """Run one backend's recognition in the current runtime image.

    Imports are intentionally deferred to the selected backend seam so a
    recognition runtime never needs preparation, finalization, or unrelated
    ASR packages.
    """
    from speech_transcriber.recognition import RecognitionRunner
    from speech_transcriber.transcription.factory import create_transcriber

    prepared = load_prepared_recording(config.input_path)
    device = _recognition_device(config.device)
    return RecognitionRunner(_memory_metrics(device, config.asr_backend)).recognize(
        prepared, create_transcriber(config, device), config.asr_backend
    )


def _finalize(args: argparse.Namespace) -> int:
    """Validate artifacts, align speakers, and export the final transcript.

    The finalizer import is deferred so recognition-only runtime images never
    need finalization machinery to run ``recognize``.
    """
    from speech_transcriber.finalization import TranscriptFinalizer

    prepared = load_prepared_recording(args.prepared)
    recognition = load_asr_recognition(
        args.asr,
        expected_backend=args.backend,
        expected_normalized_audio_sha256=prepared.normalized_audio_sha256,
    )
    config = _finalize_config(args, prepared)
    result = TranscriptFinalizer(config).finalize_prepared(
        prepared,
        recognition,
        args.output,
        expected_backend=args.backend,
    )
    write_asr_result_files(recognition, result.output_directory or args.output)
    return 0


def _finalize_config(args: argparse.Namespace, prepared: PreparedRecording) -> PipelineConfig:
    """Build the finalizer configuration from a prepared artifact's identity."""
    return PipelineConfig.from_environment(
        prepared.audio.path,
        args.output,
        {
            "language": args.language or prepared.language,
            "log_level": args.log_level,
        },
    )


def _prefetch_config(args: argparse.Namespace) -> PipelineConfig:
    """Build the prefetch configuration from explicit flags only."""
    return PipelineConfig.from_environment(
        Path("."),
        Path("."),
        {
            "asr_backend": args.backend,
            "asr_model": args.model,
            "qwen_aligner_model": (
                args.qwen_aligner_model
                or os.environ.get("QWEN_ALIGNER_MODEL", DEFAULT_QWEN_ALIGNER_MODEL)
            ),
            "pyannote_model": args.pyannote_model or DEFAULT_PYANNOTE_MODEL,
            "log_level": args.log_level,
        },
    )


def _recognition_device(requested: str) -> str:
    """Resolve ``auto`` for recognition without importing PyTorch.

    An explicit ``cuda`` or ``cpu`` passes through unchanged so the dedicated
    faster-whisper image never imports Torch. ``auto`` falls back to ``cuda``
    when the CUDA runtime is present, otherwise ``cpu``.
    """
    if requested != "auto":
        return requested
    try:
        import ctypes

        ctypes.CDLL("libcudart.so")
    except OSError:
        return "cpu"
    return "cuda"


def _memory_metrics(device: str, backend: str) -> MemoryMetrics | None:
    """Return CUDA peak accounting when Torch is importable on this path.

    The CTranslate2 runtime has no Torch, so the faster-whisper backend and any
    CLI path where Torch is absent never construct the metrics adapter; the
    Transformers and NeMo runtimes gain per-backend CUDA peaks in recognition
    metadata.
    """
    if BACKEND_RUNTIMES[backend] == "ctranslate2":
        return None
    try:
        from speech_transcriber.runtime.device import TorchMemoryMetrics
    except ImportError:
        return None
    return TorchMemoryMetrics(device) if device == "cuda" else None


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if args.command == "prepare":
        return PipelineConfig.from_environment(
            args.input,
            args.output,
            {
                "working_directory": args.working_directory,
                "device": args.device,
                "pyannote_model": args.pyannote_model,
                "language": args.language,
                "num_speakers": args.num_speakers,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
                "log_level": args.log_level,
            },
        )
    if args.command == "recognize":
        return PipelineConfig.from_environment(
            args.prepared,
            args.output,
            {
                "working_directory": args.working_directory,
                "device": args.device,
                "asr_backend": args.backend,
                "asr_model": args.model,
                "qwen_aligner_model": args.qwen_aligner_model,
                "parakeet_segment_duration": args.parakeet_segment_duration,
                "parakeet_segment_overlap": args.parakeet_segment_overlap,
                "qwen_segment_duration": args.qwen_segment_duration,
                "qwen_segment_overlap": args.qwen_segment_overlap,
                "nemotron_num_lookahead_tokens": args.nemotron_num_lookahead_tokens,
                "voxtral_delay_ms": args.voxtral_delay_ms,
                "voxtral_timestamp_offset_tokens": args.voxtral_timestamp_offset_tokens,
                "faster_whisper_compute_type": args.faster_whisper_compute_type,
                "canary_chunk_duration_seconds": args.canary_chunk_duration,
                "language": args.language,
                "log_level": args.log_level,
            },
        )
    raise AssertionError(f"unhandled prepared-artifact command: {args.command}")


def _prefetch(config: PipelineConfig) -> None:
    """Download selected ASR and pyannote model repositories for offline use.

    Pyannote access conditions must already have been accepted and HF_TOKEN must
    be present when its gated model is downloaded.
    """
    from huggingface_hub import snapshot_download

    logging.getLogger(__name__).info("prefetching ASR model")
    snapshot_download(config.resolved_asr_model)
    if config.asr_backend == "qwen":
        logging.getLogger(__name__).info("prefetching Qwen forced aligner")
        snapshot_download(config.qwen_aligner_model)
    logging.getLogger(__name__).info("prefetching pyannote model")
    snapshot_download(config.pyannote_model, token=True)


if __name__ == "__main__":
    sys.exit(main())