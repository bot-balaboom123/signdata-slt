"""MediaPipe refined 553-keypoint to 85-keypoint preset."""

PRESET = {
    "description": (
        "MediaPipe refined: 6 pose + 37 face + 21 left hand + 21 right hand"
    ),
    "source_keypoints": 553,
    "target_keypoints": 85,
    "indices": (
        # Pose (offset=0): shoulders, elbows, hips
        [11, 12, 13, 14, 23, 24]
        # Face (offset=33): 37 selected landmarks (with iris refinement)
        + [
            33, 37, 46, 47, 50, 66, 70, 72, 79, 85, 88, 94, 97,
            114, 115, 126, 166, 184, 185, 192, 205, 211, 214,
            296, 302, 309, 315, 318, 324, 327, 344, 356,
            395, 419, 430, 501, 506,
        ]
        # Left hand (offset=511): all 21
        + list(range(511, 532))
        # Right hand (offset=532): all 21
        + list(range(532, 553))
    ),
}

__all__ = ["PRESET"]
