"""Person detection ABC and factory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    """A single person detection."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class PersonDetector(ABC):
    """Abstract base class for person detection backends."""

    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect persons in a batch of frames.

        Args:
            frames: List of BGR frames (np.ndarray).

        Returns:
            For each frame, a list of Detection objects.
        """
        ...

    def close(self) -> None:
        """Release resources. Override if needed."""
        pass


def create_detector(detection_type: str, detection_config) -> PersonDetector:
    """Factory: create a detector from the detection type and config.

    Args:
        detection_type: One of "yolo", "mmdet", "mediapipe", "null".
        detection_config: Typed config model (or None for null).

    Returns:
        A PersonDetector instance.
    """
    if detection_type == "null":
        from .null import NullDetector

        return NullDetector()
    if detection_type == "yolo":
        from .yolo import YOLODetector

        return YOLODetector(detection_config)
    if detection_type == "mmdet":
        from .mmdet import MMDetDetector

        return MMDetDetector(detection_config)
    if detection_type == "mediapipe":
        from .mediapipe import MediaPipeDetector

        return MediaPipeDetector(detection_config)
    raise ValueError(f"Unknown detection type: {detection_type!r}")


__all__ = ["Detection", "PersonDetector", "create_detector"]
