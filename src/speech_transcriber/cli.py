"""Command-line interface for local and container execution."""

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
    COMPARE_BACKENDS,
    DEFAULT_PYANNOTE_MODEL,
    DEFAULT_QWEN_ALIGNER_MODEL,
    FASTER_WHISPER_COMPUTE_TYPES,
    PipelineConfig,
)
from speech_transcriber.errors import TranscriberError
from speech_transcriber.finalization import TranscriptFinalizer
from speech_transcriber.prepared import (
    load_prepared_recording,
    write_prepared_recording,
)

if TYPE_CHECKING:
    from speech_transcriber.pipeline import TranscriptionPipeline


def create_default_pipeline(config: PipelineConfig) -> TranscriptionPipeline:
    """Lazily import the ML-oriented runtime pipeline for non-finalization commands."""
    from speech_transcriber.pipeline import create_default_pipeline as create_pipeline

    return create_pipeline(config)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""
    parser = argparse.ArgumentParser(prog="speech-transcriber")
    commands = parser.add_subparsers(dest="command", required=True)
    transcribe = commands.add_parser("transcribe", help="transcribe one audio recording")
    _add_runtime_options(transcribe, include_asr=True)
    prepare = commands.add_parser(
        "prepare", help="normalize and diarize a recording for independent ASR tasks"
    )
    _add_prepare_options(prepare)
    transcribe_prepared = commands.add_parser(
        "transcribe-prepared", help="transcribe a prepared recording with one ASR backend"
    )
    _add_transcribe_prepared_options(transcribe_prepared)
    recognize_prepared = commands.add_parser(
        "recognize-prepared", help="recognize a prepared recording and write portable ASR output"
    )
    _add_transcribe_prepared_options(recognize_prepared)
    finalize_prepared = commands.add_parser(
        "finalize-prepared", help="build a transcript from prepared and ASR artifacts"
    )
    _add_finalize_prepared_options(finalize_prepared)
    compare = commands.add_parser(
        "compare", help="run several ASR backends against one prepared recording"
    )
    _add_runtime_options(compare, include_asr=False)
    compare.add_argument(
        "--models",
        default=",".join(COMPARE_BACKENDS),
        help=(
            "comma-separated generic-runtime ASR backends: parakeet, primeline, qwen, "
            "nemotron, voxtral; use Argo for heterogeneous runtime comparisons"
        ),
    )
    prefetch = commands.add_parser(
        "prefetch-models", help="download models into configured Hugging Face cache"
    )
    prefetch.add_argument("--asr", choices=ASR_BACKENDS, default="parakeet")
    prefetch.add_argument("--asr-model")
    prefetch.add_argument("--qwen-aligner-model")
    prefetch.add_argument("--pyannote-model", default=DEFAULT_PYANNOTE_MODEL)
    return parser


def _add_runtime_options(parser: argparse.ArgumentParser, include_asr: bool) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    if include_asr:
        parser.add_argument(
            "--asr",
            choices=ASR_BACKENDS,
            help="ASR backend: parakeet, primeline, qwen, nemotron, voxtral, "
            "faster-whisper, canary, granite",
        )
        parser.add_argument("--asr-model", help="Hugging Face model ID or local model directory")
    parser.add_argument("--qwen-aligner-model", help="Qwen forced-aligner model ID or local path")
    parser.add_argument("--pyannote-model")
    parser.add_argument("--parakeet-segment-duration", type=float)
    parser.add_argument("--parakeet-segment-overlap", type=float)
    parser.add_argument("--qwen-segment-duration", type=float)
    parser.add_argument("--qwen-segment-overlap", type=float)
    parser.add_argument("--granite-segment-duration", type=float)
    parser.add_argument("--granite-segment-overlap", type=float)
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
    parser.add_argument("--language", help="ASR language locale (default: de-DE)")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_prepare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--pyannote-model")
    parser.add_argument("--language", help="recording language locale (default: de-DE)")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_transcribe_prepared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--asr", choices=ASR_BACKENDS, required=True)
    parser.add_argument("--asr-model", help="Hugging Face model ID or local model directory")
    parser.add_argument("--qwen-aligner-model", help="Qwen forced-aligner model ID or local path")
    parser.add_argument("--parakeet-segment-duration", type=float)
    parser.add_argument("--parakeet-segment-overlap", type=float)
    parser.add_argument("--qwen-segment-duration", type=float)
    parser.add_argument("--qwen-segment-overlap", type=float)
    parser.add_argument("--granite-segment-duration", type=float)
    parser.add_argument("--granite-segment-overlap", type=float)
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
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_finalize_prepared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--asr-result", type=Path, required=True)
    parser.add_argument("--expected-backend", choices=ASR_BACKENDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument(
        "--language", help="transcript language locale (default: prepared artifact language)"
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a shell-compatible status code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", None) or "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        if args.command == "prefetch-models":
            prefetch_config = PipelineConfig(
                input_path=Path("."),
                output_directory=Path("."),
                working_directory=Path("/work"),
                asr_backend=args.asr,
                asr_model=args.asr_model,
                qwen_aligner_model=(
                    args.qwen_aligner_model
                    or os.environ.get("QWEN_ALIGNER_MODEL", DEFAULT_QWEN_ALIGNER_MODEL)
                ),
            )
            _prefetch(
                prefetch_config.resolved_asr_model,
                args.pyannote_model,
                prefetch_config.qwen_aligner_model
                if args.asr == "qwen" else None,
            )
            return 0
        if args.command in {"transcribe-prepared", "recognize-prepared", "finalize-prepared"}:
            prepared = load_prepared_recording(args.prepared)
            config = _config_from_args(
                args,
                input_path=prepared.audio.path,
                language=args.language or prepared.language,
            )
        else:
            config = _config_from_args(args)
        logging.getLogger(__name__).setLevel(config.log_level)
        if args.command == "finalize-prepared":
            recognition = load_asr_recognition(
                args.asr_result,
                expected_backend=args.expected_backend,
                expected_normalized_audio_sha256=prepared.normalized_audio_sha256,
            )
            result = TranscriptFinalizer(config).finalize_prepared(
                prepared,
                recognition,
                config.output_directory,
                expected_backend=args.expected_backend,
            )
            write_asr_result_files(recognition, result.output_directory or config.output_directory)
            return 0

        if args.command == "recognize-prepared":
            from speech_transcriber.recognition import RecognitionRunner
            from speech_transcriber.transcription.factory import create_transcriber

            recognition = RecognitionRunner().recognize(
                prepared,
                create_transcriber(config, _recognition_device(config.device)),
                config.asr_backend,
            )
            write_asr_recognition(recognition, config.output_directory)
            return 0

        if args.command == "compare":
            from speech_transcriber.comparison import ASRComparisonRunner

            models = [model.strip() for model in args.models.split(",") if model.strip()]
            if len(models) != len(set(models)):
                raise ValueError("--models must not contain duplicate ASR backends")
            unsupported = sorted(set(models) - set(COMPARE_BACKENDS))
            if unsupported:
                raise ValueError(
                    "--models only supports the generic runtime; use Argo for: "
                    + ", ".join(unsupported)
                )
            pipeline = create_default_pipeline(config)
            ASRComparisonRunner(pipeline).run(models, config.output_directory)
        else:
            pipeline = create_default_pipeline(config)
            if args.command == "prepare":
                prepared = pipeline.prepare()
                try:
                    write_prepared_recording(prepared, config.output_directory)
                finally:
                    pipeline.cleanup(prepared)
            elif args.command == "transcribe-prepared":
                recognition = pipeline.recognize_prepared(
                    prepared,
                    pipeline.transcriber_factory(),
                    config.asr_backend,
                )
                result = pipeline.finalize_prepared(prepared, recognition, config.output_directory)
                write_asr_result_files(
                    recognition, result.output_directory or config.output_directory
                )
            else:
                pipeline.run()
        return 0
    except (TranscriberError, OSError, RuntimeError, ValueError) as error:
        logging.getLogger(__name__).error("pipeline failed: %s", error)
        return 1


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


def _config_from_args(
    args: argparse.Namespace,
    *,
    input_path: Path | None = None,
    language: str | None = None,
) -> PipelineConfig:
    overrides = {
        "working_directory": args.working_directory,
        "device": getattr(args, "device", None),
        "asr_backend": getattr(args, "asr", None),
        "asr_model": getattr(args, "asr_model", None),
        "qwen_aligner_model": getattr(args, "qwen_aligner_model", None),
        "pyannote_model": getattr(args, "pyannote_model", None),
        "parakeet_segment_duration": getattr(args, "parakeet_segment_duration", None),
        "parakeet_segment_overlap": getattr(args, "parakeet_segment_overlap", None),
        "qwen_segment_duration": getattr(args, "qwen_segment_duration", None),
        "qwen_segment_overlap": getattr(args, "qwen_segment_overlap", None),
        "granite_segment_duration": getattr(args, "granite_segment_duration", None),
        "granite_segment_overlap": getattr(args, "granite_segment_overlap", None),
        "nemotron_num_lookahead_tokens": getattr(args, "nemotron_num_lookahead_tokens", None),
        "voxtral_delay_ms": getattr(args, "voxtral_delay_ms", None),
        "voxtral_timestamp_offset_tokens": getattr(args, "voxtral_timestamp_offset_tokens", None),
        "faster_whisper_compute_type": getattr(args, "faster_whisper_compute_type", None),
        "canary_chunk_duration_seconds": getattr(args, "canary_chunk_duration", None),
        "language": language if language is not None else getattr(args, "language", None),
        "num_speakers": getattr(args, "num_speakers", None),
        "min_speakers": getattr(args, "min_speakers", None),
        "max_speakers": getattr(args, "max_speakers", None),
        "keep_intermediate_files": getattr(args, "keep_intermediate", False),
        "log_level": getattr(args, "log_level", None),
    }
    return PipelineConfig.from_environment(input_path or args.input, args.output, overrides)


def _prefetch(asr_model: str, pyannote_model: str, qwen_aligner_model: str | None = None) -> None:
    """Download selected ASR and pyannote model repositories for offline use.

    Pyannote access conditions must already have been accepted and HF_TOKEN must
    be present when its gated model is downloaded.
    """
    from huggingface_hub import snapshot_download

    logging.getLogger(__name__).info("prefetching ASR model")
    snapshot_download(asr_model)
    if qwen_aligner_model:
        logging.getLogger(__name__).info("prefetching Qwen forced aligner")
        snapshot_download(qwen_aligner_model)
    logging.getLogger(__name__).info("prefetching pyannote model")
    snapshot_download(pyannote_model, token=True)


if __name__ == "__main__":
    sys.exit(main())
