"""FPS-based frame sampler."""

from .base import BaseSampler


class FPSSampler(BaseSampler):
    """Downsample source FPS to target FPS using Bresenham-like accumulation."""

    def __init__(self, src_fps: float, target_fps: float):
        self.target = min(target_fps, src_fps)
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
