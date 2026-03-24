"""Base sampler ABC."""

from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Abstract base class for frame sampling strategies."""

    @abstractmethod
    def take(self) -> bool:
        """Returns True if the current frame should be sampled."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset sampler state for a new video segment."""
        ...
