"""MMPose COCO WholeBody 133-keypoint to 85-keypoint preset."""

PRESET = {
    "description": (
        "COCO WholeBody 133 to 85 selected upper body, face, and hand keypoints"
    ),
    "source_keypoints": 133,
    "target_keypoints": 85,
    "indices": (
        # Upper body pose
        [5, 6, 7, 8, 11, 12]
        # Face (selected)
        + [23, 25, 27, 29, 31, 33, 35, 37, 39]
        + [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        + [52, 54, 56, 58]
        # Hands (selected + remaining)
        + [71, 73, 75, 77, 79, 81, 83, 84, 85, 86, 87, 88, 89, 90]
        + list(range(91, 133))
    ),
}

__all__ = ["PRESET"]
