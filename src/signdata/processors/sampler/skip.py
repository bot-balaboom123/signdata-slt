"""Frame-skip sampler."""

from .base import BaseSampler


class SkipSampler(BaseSampler):
    """Sample every Nth frame (simple strided sampling)."""

    def __init__(self, stride: int):
        self.n = max(int(stride), 1)
        self.count = 0

    def take(self) -> bool:
        take_now = (self.count % self.n) == 0
        self.count += 1
        return take_now

    def reset(self) -> None:
        self.count = 0
