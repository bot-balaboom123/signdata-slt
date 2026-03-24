"""Detection package exports."""

from .base import Detection, PersonDetector, create_detector
from .validation import apply_bbox_padding, single_person_check, union_bboxes


def __getattr__(name):
    if name in {"YOLO", "YOLODetector"}:
        from .yolo import YOLO, YOLODetector

        if name == "YOLO":
            return YOLO
        return YOLODetector
    if name == "MMDetDetector":
        from .mmdet import MMDetDetector

        return MMDetDetector
    if name == "MediaPipeDetector":
        from .mediapipe import MediaPipeDetector

        return MediaPipeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Detection",
    "PersonDetector",
    "create_detector",
    "single_person_check",
    "union_bboxes",
    "apply_bbox_padding",
    "YOLO",
    "YOLODetector",
    "MMDetDetector",
    "MediaPipeDetector",
]
