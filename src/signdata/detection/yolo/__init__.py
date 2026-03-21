"""YOLO-based detection backend."""

from .backend import YOLO, detect_persons_batch

__all__ = ["YOLO", "detect_persons_batch"]
