"""Device and precision selection."""

from __future__ import annotations


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to CUDA when PyTorch reports an available GPU."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but PyTorch is not installed") from None
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return "cuda" if requested == "cuda" or torch.cuda.is_available() else "cpu"


def gpu_dtype() -> object:
    """Select BF16 on capable CUDA GPUs, otherwise FP16."""
    import torch

    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def inference_dtype(device: str) -> tuple[object, str]:
    """Return the safe model precision and a stable operational label."""
    if device == "cpu":
        import torch

        return torch.float32, "float32"
    dtype = gpu_dtype()
    return dtype, "bfloat16" if str(dtype).endswith("bfloat16") else "float16"


def reset_peak_cuda_memory(device: str) -> None:
    """Reset per-run CUDA peak accounting when CUDA is the selected device."""
    if device != "cuda":
        return
    import torch

    torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory(device: str) -> tuple[int | None, int | None]:
    """Return allocated and reserved CUDA peaks for the current backend run."""
    if device != "cuda":
        return None, None
    import torch

    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


class TorchMemoryMetrics:
    """Torch-backed CUDA peak memory accounting for Transformers backends.

    This adapter is only constructed by the ML-oriented pipeline path, so the
    dedicated faster-whisper image never imports Torch for memory metrics.
    """

    def __init__(self, device: str) -> None:
        self.device = device

    def reset(self) -> None:
        reset_peak_cuda_memory(self.device)

    def peak(self) -> tuple[int | None, int | None]:
        return peak_cuda_memory(self.device)
