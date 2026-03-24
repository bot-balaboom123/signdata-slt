"""YOLO person detection backend."""

import logging
from typing import List

import numpy as np

from ..base import Detection, PersonDetector

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]


class YOLODetector(PersonDetector):
    """YOLO-based person detector using ultralytics."""

    def __init__(self, config):
        """
        Args:
            config: YOLODetectionConfig with model, device,
                    confidence_threshold, min_bbox_area.
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            )
        self.config = config
        self.model = YOLO(config.model)
        self.model.to(config.device)
        logger.info("YOLO detector loaded: %s on %s", config.model, config.device)

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        if not frames:
            return []

        results = self.model(frames, verbose=False)
        all_detections: List[List[Detection]] = []

        for i, result in enumerate(results):
            h, w = frames[i].shape[:2]
            frame_area = float(w * h)
            frame_dets = []

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                all_detections.append(frame_dets)
                continue

            class_ids = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            for j in range(len(class_ids)):
                if int(class_ids[j]) != 0:  # person class only
                    continue
                conf = float(confidences[j])
                if conf < self.config.confidence_threshold:
                    continue

                x1, y1, x2, y2 = xyxy[j]
                bbox_area = (x2 - x1) * (y2 - y1)
                if frame_area > 0 and (bbox_area / frame_area) < self.config.min_bbox_area:
                    continue

                frame_dets.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                ))

            all_detections.append(frame_dets)

        return all_detections

    def close(self) -> None:
        del self.model


__all__ = ["YOLO", "YOLODetector"]
