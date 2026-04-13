"""YOLO person detection backend."""

import logging
from typing import List

import numpy as np

from ..base import Detection, PersonDetector
from .resolver import VALID_ALIASES, resolve_yolo_model

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]
    ULTRALYTICS_VERSION = "unknown"


PERSON_CLASS_ID = 0


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


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

        if _is_cuda_device(config.device):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"YOLO device {config.device!r} requested but CUDA is unavailable. "
                    "Verify your PyTorch CUDA install and GPU visibility, or "
                    "switch detection_config.device to 'cpu'."
                )

        self.config = config
        self.use_fp16 = _is_cuda_device(config.device)

        resolved_model = resolve_yolo_model(
            config.model,
            allow_download=config.allow_download,
            weights_dir=config.weights_dir,
        )
        # Key alias_mode off the resolver's output, not the raw config, so a
        # cwd-local file named like an alias does not trip the download-error
        # branch below.
        alias_mode = resolved_model in VALID_ALIASES

        try:
            self.model = YOLO(resolved_model)
        except FileNotFoundError as e:
            if alias_mode:
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

        logger.info(
            "YOLO detector loaded: %s (resolved: %s) on %s (fp16=%s)",
            config.model, resolved_model, config.device, self.use_fp16,
        )

    def _reset_predictor(self) -> None:
        # Ultralytics reuses the existing predictor as long as the device stays
        # the same, so a previous fp16 AutoBackend would keep running even if
        # we pass half=False on a later call. Null it out to force a rebuild.
        if hasattr(self.model, "predictor"):
            self.model.predictor = None

    def _predict_chunk(self, chunk: List[np.ndarray], *, half: bool):
        return self.model.predict(
            source=chunk,
            device=self.config.device,
            verbose=False,
            half=half,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        if not frames:
            return []

        batch_size = self.config.batch_size
        all_detections: List[List[Detection]] = []

        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]

            if self.use_fp16:
                try:
                    results = self._predict_chunk(chunk, half=True)
                except Exception as fp16_exc:
                    logger.warning(
                        "FP16 inference failed (%s); resetting the Ultralytics "
                        "predictor and retrying with half=False for the rest of "
                        "this run.",
                        fp16_exc,
                    )
                    self._reset_predictor()
                    try:
                        results = self._predict_chunk(chunk, half=False)
                    except Exception as fp32_exc:
                        raise RuntimeError(
                            "YOLO inference failed with fp16, and the fp32 "
                            "retry after predictor reset also failed. "
                            f"fp16 error: {fp16_exc}; fp32 retry error: {fp32_exc}"
                        ) from fp32_exc
                    self.use_fp16 = False
            else:
                results = self._predict_chunk(chunk, half=False)

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
                    if int(class_ids[j]) != PERSON_CLASS_ID:
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
