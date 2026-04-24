"""MMDet person detection backend."""

import logging
from typing import List

import numpy as np

from .._cuda_utils import clear_cuda_cache, is_cuda_oom_error
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
        self._use_sequential = False
        logger.info(
            "MMDet detector loaded: %s on %s",
            config.det_model_config, config.device,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        if not frames:
            return []

        from mmdet.apis import inference_detector

        batch_size = self.config.batch_size
        all_detections: List[List[Detection]] = []

        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            batch_results = self._infer_chunk_adaptive(inference_detector, chunk)

            for result in batch_results:
                frame_dets = []
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

            del batch_results

        return all_detections

    def _infer_chunk_adaptive(self, inference_detector, chunk: List[np.ndarray]):
        try:
            return self._infer_chunk(inference_detector, chunk)
        except Exception as exc:
            if not is_cuda_oom_error(exc) or len(chunk) == 1:
                raise

            clear_cuda_cache(self.config.device)
            mid = max(1, len(chunk) // 2)
            logger.warning(
                "MMDet inference OOM for batch size %d; retrying as %d + %d",
                len(chunk), mid, len(chunk) - mid,
            )
            return (
                self._infer_chunk_adaptive(inference_detector, chunk[:mid])
                + self._infer_chunk_adaptive(inference_detector, chunk[mid:])
            )

    def _infer_chunk(self, inference_detector, chunk: List[np.ndarray]):
        if self._use_sequential:
            return [inference_detector(self.detector, f) for f in chunk]

        try:
            batch_results = inference_detector(self.detector, chunk)
            if not isinstance(batch_results, (list, tuple)):
                batch_results = [batch_results]
            return list(batch_results)
        except Exception as exc:
            if is_cuda_oom_error(exc):
                clear_cuda_cache(self.config.device)
                raise
            logger.warning(
                "MMDet batched inference failed; falling back to "
                "sequential mode for all subsequent frames."
            )
            self._use_sequential = True
            return [inference_detector(self.detector, f) for f in chunk]

    def close(self) -> None:
        del self.detector
        clear_cuda_cache(self.config.device)


__all__ = ["MMDetDetector"]
