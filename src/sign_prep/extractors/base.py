"""Base class for landmark extractors."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class LandmarkExtractor(ABC):
    """Abstract base class for landmark extraction."""

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame and extract landmarks.

        Args:
            frame: Input video frame (BGR format from OpenCV)

        Returns:
            Numpy array of landmarks, or None if extraction fails.
        """
        pass

    @abstractmethod
    def close(self):
        """Release resources (models, GPU memory, etc.)."""
        pass
