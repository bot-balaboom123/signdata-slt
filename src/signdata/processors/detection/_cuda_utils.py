"""Shared CUDA helpers for detection backends."""

import logging

logger = logging.getLogger(__name__)


def is_cuda_device(device: str) -> bool:
    """Return True when the configured device targets CUDA."""
    return str(device).startswith("cuda")


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True when an exception message matches a CUDA OOM failure."""
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def clear_cuda_cache(device: str) -> None:
    """Best-effort CUDA cache clear for retry paths."""
    if not is_cuda_device(device):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        logger.debug("Unable to clear CUDA cache", exc_info=True)


__all__ = ["clear_cuda_cache", "is_cuda_device", "is_cuda_oom_error"]
