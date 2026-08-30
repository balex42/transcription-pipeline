"""Transformers adapter for IBM Granite Speech 4.1 Plus word-timestamp ASR.

Granite Speech Plus is a multimodal speech LLM whose timestamp mode transcribes
speech and appends a native end-time tag after every word:

    hello [T:45] world [T:82]

Each ``[T:N]`` is the END time of the preceding word in centiseconds, emitted as
the last three digits of the count, so it wraps modulo 1000 centiseconds
(10 seconds). The parser unwraps that rollover and never invents word starts.

Granite also offers a speaker-attributed ``[Speaker N]:`` prompt mode. It is
intentionally unused: pyannote remains this pipeline's only speaker provider
and the common backend-neutral finalizer owns all speaker attribution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

from speech_transcriber.audio.segmenter import AudioSegmenter
from speech_transcriber.errors import ASROutputError, ModelLoadError
from speech_transcriber.model_cache import resolve_hf_snapshot, snapshot_revision
from speech_transcriber.models import ASRWord, NormalizedAudio, RuntimeProvenance
from speech_transcriber.runtime.device import inference_dtype
from speech_transcriber.transcription.base import Transcriber, TranscriberCapabilities
from speech_transcriber.transcription.segments import reconcile_segment_end_words

LOGGER = logging.getLogger(__name__)

GRANITE_TIMESTAMP_PROMPT = (
    "<|audio|> Timestamps: Transcribe the speech. "
    "After each word, add a timestamp tag showing the end time in centiseconds, "
    "e.g. hello [T:45] world [T:82]"
)
GRANITE_TIMESTAMP_TAG = re.compile(r"\[T:(\d+)\]")
GRANITE_SILENCE_MARKER = "_"
GRANITE_MAX_NEW_TOKENS = 4096
GRANITE_ROLLOVER_CENTISECONDS = 1000


class _GraniteTokenizer(Protocol):
    def apply_chat_template(
        self, conversation: list[dict[str, object]], **kwargs: object
    ) -> object: ...


class _GraniteProcessor(Protocol):
    tokenizer: _GraniteTokenizer

    def __call__(self, *args: object, **kwargs: object) -> Any: ...

    def batch_decode(self, token_ids: object, **kwargs: object) -> list[str]: ...


class _GraniteModel(Protocol):
    def to(self, device: str) -> object: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> Any: ...


class GraniteTranscriber(Transcriber):
    """Transcribe with Granite's native word-end timestamps in fixed segments.

    One invariant segmented path serves every recording length: the model is
    loaded once, all overlapping segments run sequentially through the same
    timestamp-mode prompt, and segment-local words are rebased and reconciled
    with their native end-only times preserved. Speaker attribution is
    exclusively downstream pyannote work handled by the common finalizer.
    """

    capabilities = TranscriberCapabilities(False, True, False, False)

    def __init__(
        self,
        model: str,
        device: str,
        segment_duration: float = 180.0,
        segment_overlap: float = 15.0,
    ) -> None:
        self.model_reference = model
        self.device = device
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        # Resolved lazily by the loader so factory construction never imports
        # PyTorch, mirroring the other Transformers backends' lazy loading.
        self.dtype_name = "granite-default"
        self._model: _GraniteModel | None = None
        self._processor: _GraniteProcessor | None = None
        self._segmenter = AudioSegmenter(segment_duration, segment_overlap)
        self.backend_metrics: dict[str, float] = {}
        self.backend_models: dict[str, str] = {}
        self.backend_configuration: dict[str, str | int | float | bool | None] = {
            "timestamp_mode": "word_end_centiseconds",
            "timestamp_rollover_centiseconds": GRANITE_ROLLOVER_CENTISECONDS,
            "segment_duration_seconds": segment_duration,
            "segment_overlap_seconds": segment_overlap,
            "max_new_tokens": GRANITE_MAX_NEW_TOKENS,
            "speaker_attribution": False,
            "forced_alignment": False,
            "do_sample": False,
        }
        self.runtime_provenance = RuntimeProvenance(
            name="transformers",
            version="unknown",
            components={"torch": "unknown", "peft": "unknown"},
        )

    def load(self) -> None:
        """Load processor and model from one resolved local snapshot."""
        self._load()

    def transcribe(self, audio: NormalizedAudio) -> list[ASRWord]:
        """Segment, run every segment through one timestamp-mode prompt, reconcile."""
        model, processor = self._load()
        segments = self._segmenter.segment(audio)
        self.backend_metrics["segments_processed"] = float(len(segments))
        words_by_segment: dict[int, list[ASRWord]] = {}
        rollovers = 0
        lexical_count = 0
        silence_count = 0
        timestamp_count = 0
        try:
            for segment in segments:
                text = self._generate_segment(processor, model, segment)
                metadata = parse_granite_timestamp_words(text)
                words_by_segment[segment.index] = metadata.words
                rollovers += metadata.rollover_count
                lexical_count += len(metadata.words)
                silence_count += metadata.silence_marker_count
                timestamp_count += metadata.timestamp_count
        except ASROutputError:
            raise
        except Exception as error:
            raise ASROutputError(
                "Granite recognition failed for "
                f"{audio.path} with model {self.model_reference}: {error}"
            ) from error
        words = reconcile_segment_end_words(segments, words_by_segment)
        self.backend_metrics["generated_word_count"] = float(lexical_count)
        self.backend_metrics["timestamp_tags_decoded"] = float(timestamp_count)
        self.backend_metrics["timestamp_rollovers"] = float(rollovers)
        self.backend_metrics["silence_markers_ignored"] = float(silence_count)
        self.backend_metrics["word_count"] = float(len(words))
        return words

    def _generate_segment(
        self, processor: _GraniteProcessor, model: Any, segment: Any
    ) -> str:
        """Run timestamp-mode generation for one PCM segment and decode its text."""
        import torch

        dtype, _ = inference_dtype(self.device)
        # The <|audio|> placeholder inside the prompt text is expanded by the
        # processor against the passed audio, following the official Granite
        # Speech chat-template transcription recipe.
        conversation: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GRANITE_TIMESTAMP_PROMPT},
                ],
            }
        ]
        prompt = _apply_chat_template(processor, conversation)
        inputs = processor(
            prompt,
            audio=[segment.audio],
            sampling_rate=segment.sample_rate,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device, dtype=dtype)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=GRANITE_MAX_NEW_TOKENS,
                return_dict_in_generate=True,
            )
        return _decode_granite_output(processor, inputs, output)

    def _load(self) -> tuple[_GraniteModel, _GraniteProcessor]:
        """Load from the resolved cache snapshot; never contact the Hub.

        The configured reference is resolved to one deterministic local
        snapshot directory before any Transformers call, then the exact same
        local path is passed to both the processor and the model so air-gapped
        recognition reads the very same files.
        """
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            auto_processor, model_class = _transformers_factories()
            model_path = resolve_granite_model_path(self.model_reference)
            revision = snapshot_revision(Path(model_path))
            LOGGER.info(
                "loading Granite from cached snapshot revision %.12s", revision or "unknown"
            )
            dtype, dtype_name = inference_dtype(self.device)
            self.dtype_name = dtype_name
            processor = cast(
                _GraniteProcessor,
                auto_processor.from_pretrained(  # type: ignore[attr-defined]
                    str(model_path), local_files_only=True, trust_remote_code=False
                ),
            )
            model = cast(
                _GraniteModel,
                model_class.from_pretrained(  # type: ignore[attr-defined]
                    str(model_path), dtype=dtype, local_files_only=True, trust_remote_code=False
                ),
            )
            model.to(self.device)
            model.eval()
            self._model, self._processor = model, processor
            self.backend_models["model_snapshot"] = revision or "unknown"
            self.runtime_provenance = RuntimeProvenance(
                name="transformers",
                version=_package_version("transformers"),
                components={
                    "torch": _package_version("torch"),
                    "peft": _package_version("peft"),
                },
            )
            return model, processor
        except Exception as error:
            raise ModelLoadError(
                f"could not load Granite model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop model state so one GPU can serve the next backend safely."""
        self._model = None
        self._processor = None


def _transformers_factories() -> tuple[type[object], type[object]]:
    """Import the two Transformers factories lazily for offline Granite use."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    return AutoProcessor, AutoModelForSpeechSeq2Seq


def resolve_granite_model_path(model_reference: str) -> str:
    """Resolve the configured reference to one local snapshot directory.

    An existing absolute local directory is used directly for mirrored
    deployments. A repository ID resolves through the shared Hugging Face
    cache helper, which honors ``refs/main`` and fails instead of guessing
    when the cache is ambiguous or absent. There is no online fallback.
    """
    if model_reference.startswith("/"):
        local = Path(model_reference)
        if local.is_dir():
            return str(local)
        raise ModelLoadError(f"configured Granite model path {model_reference!r} does not exist")
    return str(resolve_hf_snapshot(model_reference, subject="Granite model"))


@dataclass
class GraniteTimestampMetadata:
    """Parsed Granite timestamp-mode output with diagnostic counters."""

    words: list[ASRWord]
    timestamp_count: int
    rollover_count: int
    silence_marker_count: int


@dataclass
class _PendingWord:
    """One lexical token waiting for its closing timestamp tag."""

    text: str


def parse_granite_timestamp_words(text: str) -> GraniteTimestampMetadata:
    """Parse one segment of Granite timestamp-mode text into segment-local words.

    ``[T:N]`` tags close the preceding lexical word with its END time in
    centiseconds. Only the last three digits are emitted, so the count wraps
    every 1000 centiseconds (10 seconds) and is unwrapped by tracking the
    accumulated rollover count. ``_`` marks are Granite's silence markers:
    timing records that are never transcript words and never close a lexical
    word. Lexical text without a following valid timestamp is model output
    corruption and raises ``ASROutputError`` instead of guessing timestamps.
    """
    words: list[ASRWord] = []
    pending: _PendingWord | None = None
    # Absolute end time of the previous word in centiseconds.
    previous_end_cs = -1
    rollover_wraps = 0
    silence_markers = 0
    # True while a `_` marker awaits its own closing [T:N] tag.
    pending_silence = False

    for token in re.split(r"(\[T:\d+\])", text):
        tag_match = GRANITE_TIMESTAMP_TAG.fullmatch(token)
        if tag_match is not None:
            raw_centiseconds = int(tag_match.group(1))
            absolute_end, rollover_wraps = _unwrap_timestamp(
                raw_centiseconds, previous_end_cs, rollover_wraps
            )
            if pending is None:
                if not pending_silence:
                    # A tag with no pending record whatsoever is a leading or
                    # duplicated pause; keep it as timing information only.
                    silence_markers += 1
                previous_end_cs = max(previous_end_cs, absolute_end)
                pending_silence = False
                continue
            if absolute_end < previous_end_cs:
                raise ASROutputError(
                    "Granite word timestamps are not monotonic after rollover "
                    f"unwrapping: '{pending.text}' ends at {absolute_end / 100:.2f}s "
                    f"before {previous_end_cs / 100:.2f}s"
                )
            words.append(ASRWord(text=pending.text, start=None, end=absolute_end / 100))
            previous_end_cs = absolute_end
            pending = None
            continue
        for piece in token.split():
            if piece == GRANITE_SILENCE_MARKER:
                # A silence marker is a timing record, not a word; it must not
                # replace or close the pending lexical word.
                silence_markers += 1
                pending_silence = True
                continue
            if pending is not None:
                raise ASROutputError(
                    f"Granite lexical term '{pending.text}' has no closing timestamp tag"
                )
            pending = _PendingWord(piece)

    if pending is not None:
        raise ASROutputError(f"Granite lexical term '{pending.text}' has no closing timestamp tag")
    return GraniteTimestampMetadata(
        words=words,
        timestamp_count=len(words),
        rollover_count=rollover_wraps,
        silence_marker_count=silence_markers,
    )


def _unwrap_timestamp(
    raw_centiseconds: int, previous_end_cs: int, rollover_wraps: int
) -> tuple[int, int]:
    """Unwrap Granite's modulo-1000 centisecond tag into an absolute time.

    A tag smaller than the low three digits of the previous end time means the
    counter wrapped past 10 seconds and the wrap count increments. Example:
    ``word1 [T:950] word2 [T:20]`` means 9.50s then 10.20s, not 0.20s.
    """
    wraps = rollover_wraps
    if previous_end_cs >= 0 and raw_centiseconds < previous_end_cs % GRANITE_ROLLOVER_CENTISECONDS:
        wraps += 1
    return wraps * GRANITE_ROLLOVER_CENTISECONDS + raw_centiseconds, wraps


def _apply_chat_template(
    processor: _GraniteProcessor, conversation: list[dict[str, object]]
) -> str:
    """Render the timestamp prompt through the processor's official template."""
    prompt = processor.tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    if not isinstance(prompt, str) or not prompt:
        raise ASROutputError("Granite chat template did not render a text prompt")
    return prompt


def _decode_granite_output(processor: _GraniteProcessor, inputs: Any, output: Any) -> str:
    """Strip the prompt from generated sequences and decode the tail to text."""
    sequences = getattr(output, "sequences", None)
    if sequences is None:
        raise ASROutputError("Granite generation did not return output sequences")
    if isinstance(inputs, dict):
        prompt_ids = inputs.get("input_ids", None)
    else:
        prompt_ids = getattr(inputs, "input_ids", None)
    if prompt_ids is None:
        raise ASROutputError("Granite processor did not return prompt input IDs")
    prompt_length = int(prompt_ids.shape[-1])
    generated = sequences[:, prompt_length:]
    if not sequences.numel() or generated.shape[-1] == 0:
        raise ASROutputError("Granite generated no tokens for an audio segment")
    if generated.shape[-1] >= GRANITE_MAX_NEW_TOKENS:
        raise ASROutputError(
            "Granite hit the bounded max_new_tokens limit of "
            f"{GRANITE_MAX_NEW_TOKENS}; refusing to emit a truncated transcript"
        )
    decoded = list(processor.batch_decode(generated, skip_special_tokens=True))
    if not decoded:
        raise ASROutputError("Granite decoding produced no text")
    return decoded[0]


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"