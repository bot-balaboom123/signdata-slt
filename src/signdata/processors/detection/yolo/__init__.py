"""YOLO detection backend package."""

from .backend import YOLO, YOLODetector
from .helpers import detect_persons_batch

__all__ = ["YOLO", "YOLODetector", "detect_persons_batch"]
