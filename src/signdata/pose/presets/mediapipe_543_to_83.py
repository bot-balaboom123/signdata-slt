"""MediaPipe unrefined 543-keypoint to 83-keypoint preset."""

PRESET = {
    "description": (
        "MediaPipe unrefined: 6 pose + 35 face + 21 left hand + 21 right hand"
    ),
    "source_keypoints": 543,
    "target_keypoints": 83,
    "indices": (
        # Pose (offset=0): shoulders, elbows, hips
        [11, 12, 13, 14, 23, 24]
        # Face (offset=33): 35 selected landmarks (no iris)
        + [
            33, 37, 46, 47, 50, 66, 70, 72, 79, 85, 88, 94, 97,
            114, 115, 126, 166, 184, 185, 192, 205, 211, 214,
            296, 302, 309, 315, 318, 324, 327, 344, 356,
            395, 419, 430,
        ]
        # Left hand (offset=501): all 21
        + list(range(501, 522))
        # Right hand (offset=522): all 21
        + list(range(522, 543))
    ),
}

__all__ = ["PRESET"]
