"""Detection backends."""

from .yolo import YOLO, detect_persons_batch

__all__ = ["YOLO", "detect_persons_batch"]
