"""Base class for landmark extractors."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class LandmarkExtractor(ABC):
    """Abstract base class for landmark extraction."""

    # Whether the extractor supports true batch inference (GPU batching)
    supports_batch_inference: bool = False

    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, bbox: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Process a single frame and extract landmarks.

        Args:
            frame: Input video frame (BGR format from OpenCV).
            bbox: Optional bounding box array of shape (N, 4) with
                [x1, y1, x2, y2] in pixel coordinates, provided by
                an upstream detector. Backends that perform their own
                detection (e.g. MediaPipe Holistic) may ignore this.

        Returns:
            Numpy array of landmarks, or None if extraction fails.
        """
        pass

    def process_batch(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[Optional[np.ndarray]]] = None,
        fallback_on_error: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """Process a batch of frames and extract landmarks.

        Default implementation processes frames sequentially.
        Subclasses may override for true batch inference.

        Args:
            frames: List of input video frames (BGR format from OpenCV).
            bboxes: Optional per-frame bounding boxes from upstream
                detector. Each element is an array of shape (N, 4) or
                None. Backends that perform their own detection may
                ignore this.
            fallback_on_error: If True, continue processing remaining frames
                when one fails. If False, raise the exception.

        Returns:
            List of numpy arrays of landmarks (or None for failed frames).
        """
        results = []
        for i, frame in enumerate(frames):
            try:
                bbox = bboxes[i] if bboxes else None
                landmarks = self.process_frame(frame, bbox=bbox)
                results.append(landmarks)
            except Exception:
                if fallback_on_error:
                    results.append(None)
                else:
                    raise
        return results

    @abstractmethod
    def close(self):
        """Release resources (models, GPU memory, etc.)."""
        pass

    # Number of landmarks this extractor produces. Set in subclass __init__.
    num_landmarks: int = 0


def create_estimator(pose_type: str, pose_config) -> LandmarkExtractor:
    """Factory: create a pose estimator from the pose type and config.

    Args:
        pose_type: One of "mediapipe", "mmpose".
        pose_config: Typed config model (MediaPipePoseConfig or MMPosePoseConfig).

    Returns:
        A LandmarkExtractor instance.
    """
    if pose_type == "mediapipe":
        from .mediapipe import MediaPipeExtractor

        # Bridge new config to legacy ExtractorConfig expected by MediaPipeExtractor
        from ..config.schema import MediaPipePoseConfig
        cfg = pose_config if isinstance(pose_config, MediaPipePoseConfig) else MediaPipePoseConfig(**pose_config)

        # MediaPipeExtractor expects an ExtractorConfig-like object
        class _Cfg:
            name = "mediapipe"
            model_complexity = cfg.model_complexity
            min_detection_confidence = cfg.min_detection_confidence
            min_tracking_confidence = cfg.min_tracking_confidence
            refine_face_landmarks = cfg.refine_face_landmarks
            batch_size = cfg.batch_size

        return MediaPipeExtractor(_Cfg())

    elif pose_type == "mmpose":
        from .mmpose import MMPoseExtractor
        from mmpose.apis import init_model

        pose_estimator = init_model(
            pose_config.pose_model_config,
            pose_config.pose_model_checkpoint,
            device=pose_config.device,
        )
        pose_estimator.cfg.model.test_cfg.mode = "3d"

        class _Cfg:
            name = "mmpose"
            bbox_threshold = pose_config.bbox_threshold
            keypoint_threshold = pose_config.keypoint_threshold
            add_visible = True
            batch_size = pose_config.batch_size

        return MMPoseExtractor(
            config=_Cfg(),
            detector=None,
            pose_estimator=pose_estimator,
        )

    else:
        raise ValueError(f"Unknown pose type: {pose_type!r}")
