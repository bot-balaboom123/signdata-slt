"""Pose package exports.

Keep this package lightweight so importing ``signdata.pose.presets`` does
not pull in heavy backend dependencies such as MediaPipe or MMPose.
Backend registration happens through explicit subpackage imports.
"""

from .base import LandmarkExtractor
from .presets import KEYPOINT_PRESETS, list_presets, resolve_keypoint_indices


def __getattr__(name):
    if name == "MediaPipeExtractor":
        from .mediapipe import MediaPipeExtractor
        return MediaPipeExtractor
    if name in {"MMPoseExtractor", "MultiPersonDetected"}:
        from .mmpose import MMPoseExtractor, MultiPersonDetected
        if name == "MMPoseExtractor":
            return MMPoseExtractor
        return MultiPersonDetected
    raise AttributeError(f"module 'signdata.pose' has no attribute {name!r}")

__all__ = [
    "LandmarkExtractor",
    "MediaPipeExtractor",
    "MMPoseExtractor",
    "MultiPersonDetected",
    "KEYPOINT_PRESETS",
    "list_presets",
    "resolve_keypoint_indices",
]
