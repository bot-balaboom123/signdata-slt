"""YOLO detection backend package."""

from .backend import YOLO, YOLODetector
from .resolver import VALID_ALIASES, is_valid_alias, resolve_yolo_model

__all__ = ["YOLO", "VALID_ALIASES", "YOLODetector", "is_valid_alias", "resolve_yolo_model"]
