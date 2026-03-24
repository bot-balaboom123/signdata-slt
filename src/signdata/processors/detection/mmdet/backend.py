"""MMDet person detection backend."""

import logging
from typing import List

import numpy as np

from ..base import Detection, PersonDetector

logger = logging.getLogger(__name__)


class MMDetDetector(PersonDetector):
    """MMDetection-based person detector.

    Uses the same RTMDet models that MMPose ships for top-down pose estimation.
    """

    def __init__(self, config):
        """
        Args:
            config: MMDetDetectionConfig with det_model_config,
                    det_model_checkpoint, device.
        """
        from mmdet.apis import init_detector
        from mmpose.utils import adapt_mmdet_pipeline

        self.config = config
        self.detector = init_detector(
            config.det_model_config,
            config.det_model_checkpoint,
            device=config.device,
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        logger.info(
            "MMDet detector loaded: %s on %s",
            config.det_model_config, config.device,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        from mmdet.apis import inference_detector

        all_detections: List[List[Detection]] = []

        for frame in frames:
            result = inference_detector(self.detector, frame)
            frame_dets = []

            # MMDet returns DetDataSample with pred_instances
            pred = result.pred_instances
            if pred is not None and len(pred) > 0:
                bboxes = pred.bboxes.cpu().numpy()
                scores = pred.scores.cpu().numpy()
                labels = pred.labels.cpu().numpy()

                for j in range(len(labels)):
                    if int(labels[j]) != 0:  # person class
                        continue
                    x1, y1, x2, y2 = bboxes[j]
                    frame_dets.append(Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(scores[j]),
                    ))

            all_detections.append(frame_dets)

        return all_detections

    def close(self) -> None:
        del self.detector


__all__ = ["MMDetDetector"]
