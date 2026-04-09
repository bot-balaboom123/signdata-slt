"""YOLO person detection backend."""

import logging
from typing import List

import numpy as np

from ..base import Detection, PersonDetector
from .resolver import is_valid_alias, resolve_yolo_model

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]
    ULTRALYTICS_VERSION = "unknown"


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

        if str(config.device).startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"YOLO device {config.device!r} requested but CUDA is unavailable. "
                    "Verify your PyTorch CUDA install and GPU visibility, or "
                    "switch detection_config.device to 'cpu'."
                )

        self.config = config

        # Resolve model alias / path before loading (read-only, no global mutation)
        resolved_model = resolve_yolo_model(
            config.model,
            allow_download=config.allow_download,
            weights_dir=config.weights_dir,
        )
        alias_mode = is_valid_alias(config.model)  # accepts both stem and .pt

        try:
            self.model = YOLO(resolved_model)
        except FileNotFoundError as e:
            if alias_mode:
                # A valid alias that failed to load is a download/cache problem,
                # not a user-provided path error.
                raise RuntimeError(
                    f"YOLO model {config.model!r} is a valid alias but could not "
                    f"be loaded. This usually means the download failed or the "
                    f"Ultralytics cache is missing/unwritable. Check your internet "
                    f"connection and cache directory permissions."
                ) from e
            raise FileNotFoundError(
                f"YOLO weights not found: {config.model!r}. Provide an existing "
                "local weights path or use a model alias supported by "
                f"ultralytics {ULTRALYTICS_VERSION}."
            ) from e
        except Exception as e:
            msg = str(e).lower()
            if "download" in msg or "connection" in msg or "url" in msg:
                raise RuntimeError(
                    f"Failed to download YOLO model {config.model!r}. "
                    "Check your internet connection, or provide a local "
                    "weights path instead."
                ) from e
            raise

        self.model.to(config.device)
        logger.info(
            "YOLO detector loaded: %s (resolved: %s) on %s",
            config.model, resolved_model, config.device,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        if not frames:
            return []

        batch_size = self.config.batch_size
        all_detections: List[List[Detection]] = []

        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            results = self.model(chunk, verbose=False)

            for i, result in enumerate(results):
                h, w = chunk[i].shape[:2]
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
