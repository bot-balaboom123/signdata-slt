"""Frame sampling strategies."""

from typing import Optional

from .base import BaseSampler
from .fps import FPSSampler
from .skip import SkipSampler
from .read import read_sampled_frames


def create_sampler(
    sample_rate: Optional[float] = None,
    source_fps: Optional[float] = None,
) -> BaseSampler:
    """Factory: create a frame sampler.

    Sampling rules:
      - ``None`` => keep native FPS
      - ``0 < sample_rate < 1`` => keep that ratio of source frames
      - ``sample_rate >= 1`` => downsample to that absolute FPS

    Args:
        sample_rate: Native / ratio / FPS selector.
        source_fps: Source video FPS.

    Returns:
        A BaseSampler instance.
    """
    if source_fps is not None and source_fps > 0:
        return FPSSampler(source_fps, sample_rate)
    return SkipSampler(1)


__all__ = [
    "BaseSampler",
    "FPSSampler",
    "SkipSampler",
    "create_sampler",
    "read_sampled_frames",
]
