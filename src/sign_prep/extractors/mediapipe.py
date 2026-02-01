"""MediaPipe-based holistic landmark extraction."""

from typing import Optional, List

import cv2
import mediapipe as mp
import numpy as np

from .base import LandmarkExtractor
from ..registry import register_extractor
from ..config.schema import ExtractorConfig


@register_extractor("mediapipe")
class MediaPipeExtractor(LandmarkExtractor):
    """Extracts holistic landmarks using MediaPipe.

    Extracts landmarks for pose, face, and hands. Supports optional
    keypoint reduction via index filtering.
    """

    def __init__(self, config: ExtractorConfig):
        self.pose_idx = config.pose_idx
        self.face_idx = config.face_idx
        self.hand_idx = config.hand_idx
        self.apply_reduction = config.reduction

        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=config.model_complexity,
            refine_face_landmarks=config.refine_face_landmarks,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract holistic landmarks from a single frame.

        Returns array of shape (num_keypoints, 4) with [x, y, z, visibility].
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        if self.apply_reduction:
            pose_landmarks = self._convert_landmarks_to_array(
                getattr(results.pose_landmarks, "landmark", None), self.pose_idx
            )
            face_landmarks = self._convert_landmarks_to_array(
                getattr(results.face_landmarks, "landmark", None), self.face_idx
            )
            left_hand_landmarks = self._convert_landmarks_to_array(
                getattr(results.left_hand_landmarks, "landmark", None), self.hand_idx
            )
            right_hand_landmarks = self._convert_landmarks_to_array(
                getattr(results.right_hand_landmarks, "landmark", None), self.hand_idx
            )
        else:
            pose_landmarks = self._convert_all_landmarks_to_array(
                getattr(results.pose_landmarks, "landmark", None)
            )
            face_landmarks = self._convert_all_landmarks_to_array(
                getattr(results.face_landmarks, "landmark", None)
            )
            left_hand_landmarks = self._convert_all_landmarks_to_array(
                getattr(results.left_hand_landmarks, "landmark", None)
            )
            right_hand_landmarks = self._convert_all_landmarks_to_array(
                getattr(results.right_hand_landmarks, "landmark", None)
            )

        landmark_array = np.concatenate([
            pose_landmarks,
            face_landmarks,
            left_hand_landmarks,
            right_hand_landmarks,
        ], axis=0)

        return landmark_array.astype(np.float32)

    def _convert_landmarks_to_array(
        self,
        landmarks: Optional[List],
        indices: List[int],
    ) -> np.ndarray:
        """Convert MediaPipe landmarks at specified indices to numpy array."""
        if landmarks:
            out = []
            for i in indices:
                lm = landmarks[i]
                vis = getattr(lm, "visibility", 1.0)
                out.append([lm.x, lm.y, lm.z, vis])
            return np.array(out, dtype=np.float32)
        else:
            return np.zeros((len(indices), 4), dtype=np.float32)

    def _convert_all_landmarks_to_array(
        self,
        landmarks: Optional[List],
    ) -> np.ndarray:
        """Convert all MediaPipe landmarks to numpy array."""
        if landmarks:
            out = []
            for lm in landmarks:
                vis = getattr(lm, "visibility", 1.0)
                out.append([lm.x, lm.y, lm.z, vis])
            return np.array(out, dtype=np.float32)
        else:
            return np.zeros((0, 4), dtype=np.float32)

    def close(self):
        if self.holistic is not None:
            self.holistic.close()
