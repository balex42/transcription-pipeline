"""Explicit model cleanup for sequential GPU stages."""

from __future__ import annotations

import gc
from typing import Protocol


class Releasable(Protocol):
    """A resource holding model memory."""

    def release(self) -> None:
        """Release native model references."""


def release_model(component: Releasable) -> None:
    """Release a model and return cached CUDA memory to PyTorch where possible."""
    component.release()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
