"""Pose package exports.

Keep this package lightweight so importing preset helpers does not pull in
heavy backend dependencies such as MediaPipe or MMPose. Backend registration
happens through explicit subpackage imports or lazy access via ``__getattr__``.
"""

from .base import (
    KEYPOINT_PRESETS,
    LandmarkExtractor,
    create_estimator,
    list_presets,
    resolve_keypoint_indices,
)


def __getattr__(name):
    if name == "MediaPipeExtractor":
        from .mediapipe import MediaPipeExtractor

        return MediaPipeExtractor
    if name in {"MMPoseExtractor", "MultiPersonDetected"}:
        from .mmpose import MMPoseExtractor, MultiPersonDetected

        if name == "MMPoseExtractor":
            return MMPoseExtractor
        return MultiPersonDetected
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LandmarkExtractor",
    "create_estimator",
    "MediaPipeExtractor",
    "MMPoseExtractor",
    "MultiPersonDetected",
    "KEYPOINT_PRESETS",
    "list_presets",
    "resolve_keypoint_indices",
]
