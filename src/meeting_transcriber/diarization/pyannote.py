"""Local pyannote Community-1 diarization adapter."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

from meeting_transcriber.diarization.base import Diarizer
from meeting_transcriber.errors import ModelLoadError
from meeting_transcriber.models import DiarizationSegment

LOGGER = logging.getLogger(__name__)


class PyannoteDiarizer(Diarizer):
    """Run Community-1 locally and expose its exclusive diarization track."""

    def __init__(
        self,
        model: str,
        device: str,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        self.model_reference = model
        self.device = device
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline: object | None = None

    def diarize(self, normalized_wav: Path) -> list[DiarizationSegment]:
        """Diarize once over the whole meeting using exclusive segments."""
        pipeline = self._load()
        arguments = {
            key: value
            for key, value in {
                "num_speakers": self.num_speakers,
                "min_speakers": self.min_speakers,
                "max_speakers": self.max_speakers,
            }.items()
            if value is not None
        }
        # TorchCodec 0.7 supports FFmpeg through version 7 only. Loading our
        # already-normalized PCM with SoundFile keeps pyannote independent of
        # the host's FFmpeg ABI (Fedora currently provides FFmpeg 8).
        import soundfile as sf
        import torch

        samples, sample_rate = sf.read(normalized_wav, dtype="float32", always_2d=True)
        audio_input = {
            "waveform": torch.from_numpy(samples.T.copy()),
            "sample_rate": sample_rate,
        }
        LOGGER.info("running pyannote diarization", extra={"device": self.device})
        output = pipeline(audio_input, **arguments)  # type: ignore[operator]
        annotation = output.exclusive_speaker_diarization
        raw_segments = [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        labels = {
            label: f"SPEAKER_{index:02d}"
            for index, label in enumerate(sorted({item[2] for item in raw_segments}))
        }
        return [
            DiarizationSegment(speaker=labels[speaker], start=start, end=end)
            for start, end, speaker in raw_segments
            if end > start
        ]

    def _load(self) -> object:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch

            # We always pass a preloaded waveform, so TorchCodec's host ABI
            # warning is not actionable and obscures actual model failures.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"\s*torchcodec is not installed correctly.*",
                    category=UserWarning,
                    module=r"pyannote\.audio\.core\.io",
                )
                from pyannote.audio import Pipeline

            token = os.environ.get("HF_TOKEN")
            LOGGER.info(
                "loading pyannote model",
                extra={"model": self.model_reference, "device": self.device},
            )
            pipeline = Pipeline.from_pretrained(self.model_reference, token=token)
            if pipeline is None:
                raise ModelLoadError(f"pyannote returned no pipeline for {self.model_reference}")
            if self.device == "cuda":
                pipeline.to(torch.device("cuda"))
            self._pipeline = pipeline
            LOGGER.info("loaded pyannote model")
            return pipeline
        except ModelLoadError:
            raise
        except Exception as error:
            raise ModelLoadError(
                f"could not load pyannote model {self.model_reference}: {error}"
            ) from error

    def release(self) -> None:
        """Drop pipeline references so the lifecycle manager can reclaim memory."""
        self._pipeline = None
