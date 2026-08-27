"""Command-line interface for local and container execution."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meeting_transcriber.comparison import ASRComparisonRunner
from meeting_transcriber.config import (
    ASR_BACKENDS,
    DEFAULT_PYANNOTE_MODEL,
    PipelineConfig,
)
from meeting_transcriber.errors import MeetingTranscriberError
from meeting_transcriber.pipeline import create_default_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""
    parser = argparse.ArgumentParser(prog="meeting-transcriber")
    commands = parser.add_subparsers(dest="command", required=True)
    transcribe = commands.add_parser("transcribe", help="transcribe one German meeting recording")
    _add_runtime_options(transcribe, include_asr=True)
    compare = commands.add_parser(
        "compare", help="run several ASR backends against one prepared meeting"
    )
    _add_runtime_options(compare, include_asr=False)
    compare.add_argument(
        "--models",
        default="parakeet,whisper,granite",
        help="comma-separated ASR backends: parakeet, whisper, granite",
    )
    prefetch = commands.add_parser(
        "prefetch-models", help="download models into configured Hugging Face cache"
    )
    prefetch.add_argument("--asr", choices=ASR_BACKENDS, default="parakeet")
    prefetch.add_argument("--asr-model")
    prefetch.add_argument("--pyannote-model", default=DEFAULT_PYANNOTE_MODEL)
    return parser


def _add_runtime_options(parser: argparse.ArgumentParser, include_asr: bool) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    if include_asr:
        parser.add_argument(
            "--asr", choices=ASR_BACKENDS, help="ASR backend: parakeet, whisper, granite"
        )
        parser.add_argument("--asr-model", help="Hugging Face model ID or local model directory")
    parser.add_argument("--granite-model", help="deprecated Granite-only model alias")
    parser.add_argument("--pyannote-model")
    parser.add_argument("--chunk-duration", type=float)
    parser.add_argument("--chunk-overlap", type=float)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--keep-intermediate", action="store_true")
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
            )
            _prefetch(prefetch_config.resolved_asr_model, args.pyannote_model)
            return 0
        config = _config_from_args(args)
        logging.getLogger(__name__).setLevel(config.log_level)
        pipeline = create_default_pipeline(config)
        if args.command == "compare":
            models = [model.strip() for model in args.models.split(",") if model.strip()]
            if len(models) != len(set(models)):
                raise ValueError("--models must not contain duplicate ASR backends")
            ASRComparisonRunner(pipeline).run(models, config.output_directory)
        else:
            pipeline.run()
        return 0
    except (MeetingTranscriberError, OSError, RuntimeError, ValueError) as error:
        logging.getLogger(__name__).error("pipeline failed: %s", error)
        return 1


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    overrides = {
        "working_directory": args.working_directory,
        "device": args.device,
        "asr_backend": getattr(args, "asr", None),
        "asr_model": getattr(args, "asr_model", None),
        "granite_model": args.granite_model,
        "pyannote_model": args.pyannote_model,
        "chunk_duration": args.chunk_duration,
        "chunk_overlap": args.chunk_overlap,
        "num_speakers": args.num_speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
        "keep_intermediate_files": args.keep_intermediate,
        "log_level": args.log_level,
    }
    return PipelineConfig.from_environment(args.input, args.output, overrides)


def _prefetch(asr_model: str, pyannote_model: str) -> None:
    """Download selected ASR and pyannote model repositories for offline use.

    Pyannote access conditions must already have been accepted and HF_TOKEN must
    be present when its gated model is downloaded.
    """
    from huggingface_hub import snapshot_download

    logging.getLogger(__name__).info("prefetching ASR model")
    snapshot_download(asr_model)
    logging.getLogger(__name__).info("prefetching pyannote model")
    snapshot_download(pyannote_model, token=True)


if __name__ == "__main__":
    sys.exit(main())
