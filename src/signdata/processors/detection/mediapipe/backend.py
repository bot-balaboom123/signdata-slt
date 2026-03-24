"""MediaPipe detection-only backend."""

import logging
from typing import List

import numpy as np

from ..base import Detection, PersonDetector

logger = logging.getLogger(__name__)


class MediaPipeDetector(PersonDetector):
    """Lightweight MediaPipe-based person detection.

    Uses MediaPipe's pose detection (not the full holistic pipeline)
    to locate a person in the frame.
    """

    def __init__(self, config):
        """
        Args:
            config: MediaPipeDetectionConfig with min_detection_confidence.
        """
        import mediapipe as mp
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,  # lightweight for detection only
            min_detection_confidence=config.min_detection_confidence,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        import cv2
        all_detections: List[List[Detection]] = []

        for frame in frames:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.detector.process(rgb)

            if result.pose_landmarks:
                # Compute bounding box from visible landmarks
                xs = [lm.x * w for lm in result.pose_landmarks.landmark if lm.visibility > 0.3]
                ys = [lm.y * h for lm in result.pose_landmarks.landmark if lm.visibility > 0.3]

                if xs and ys:
                    det = Detection(
                        bbox=(min(xs), min(ys), max(xs), max(ys)),
                        confidence=1.0,
                    )
                    all_detections.append([det])
                else:
                    all_detections.append([])
            else:
                all_detections.append([])

        return all_detections

    def close(self) -> None:
        self.detector.close()


__all__ = ["MediaPipeDetector"]
