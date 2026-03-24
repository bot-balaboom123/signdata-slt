"""Frame sampling strategies."""

from typing import Optional

from .base import BaseSampler
from .fps import FPSSampler
from .skip import SkipSampler
from .read import read_sampled_frames


def create_sampler(
    frame_skip: int,
    target_fps: Optional[float] = None,
    source_fps: Optional[float] = None,
) -> BaseSampler:
    """Factory: create a frame sampler.

    If target_fps and source_fps are both provided, uses FPS-based
    downsampling. Otherwise falls back to simple frame skipping.

    Args:
        frame_skip: Take every Nth frame (used as fallback).
        target_fps: Target output FPS (if available).
        source_fps: Source video FPS (needed for FPS-based sampling).

    Returns:
        A BaseSampler instance.
    """
    if target_fps is not None and source_fps is not None and source_fps > 0:
        return FPSSampler(source_fps, target_fps)
    return SkipSampler(frame_skip)


__all__ = [
    "BaseSampler",
    "FPSSampler",
    "SkipSampler",
    "create_sampler",
    "read_sampled_frames",
]
