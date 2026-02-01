"""Base dataset class."""

from abc import ABC, abstractmethod
from typing import List


class BaseDataset(ABC):
    """Abstract base class for dataset definitions."""

    name: str

    @classmethod
    @abstractmethod
    def default_config(cls) -> dict:
        """Return default config dict for this dataset."""
        pass

    @classmethod
    @abstractmethod
    def pipeline_steps(cls, mode: str) -> List[str]:
        """Return ordered list of processor names for the given mode."""
        pass
