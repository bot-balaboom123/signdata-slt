"""FPS-based frame sampler."""

from ...utils.video import resolve_effective_sample_fps
from .base import BaseSampler


class FPSSampler(BaseSampler):
    """Sample native, ratio-reduced, or absolute-FPS frames uniformly."""

    def __init__(self, src_fps: float, sample_rate: float | None):
        effective_fps = resolve_effective_sample_fps(src_fps, sample_rate)
        self.target = src_fps if effective_fps is None else min(effective_fps, src_fps)
        self.r = self.target / max(src_fps, 1e-6)
        self.acc = 0.0

    def take(self) -> bool:
        self.acc += self.r
        if self.acc >= 1.0:
            self.acc -= 1.0
            return True
        return False

    def reset(self) -> None:
        self.acc = 0.0
