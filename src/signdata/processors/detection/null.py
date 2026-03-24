"""Null detector — returns full-frame bbox for every frame."""

from typing import List

import numpy as np

from .backends import Detection, PersonDetector


class NullDetector(PersonDetector):
    """Pass-through detector that assumes the entire frame is the signer.

    Use when the pose backend handles detection internally (e.g. MediaPipe
    Holistic) or when no detection is needed.
    """

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        results = []
        for frame in frames:
            h, w = frame.shape[:2]
            det = Detection(
                bbox=(0.0, 0.0, float(w), float(h)),
                confidence=1.0,
            )
            results.append([det])
        return results
