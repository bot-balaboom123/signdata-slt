"""YOLO-specific raw detection helpers."""

from typing import List, Tuple

import numpy as np


def detect_persons_batch(
    model,
    frames_with_meta: List[Tuple[np.ndarray, int, int]],
    confidence_threshold: float,
    min_bbox_area_ratio: float,
) -> List[List[Tuple[float, float, float, float]]]:
    """Run YOLO batch inference and keep valid person boxes only."""
    if not frames_with_meta:
        return []

    images = [frame for frame, _, _ in frames_with_meta]
    results = model(images, verbose=False)

    all_bboxes: List[List[Tuple[float, float, float, float]]] = []
    for frame_index, result in enumerate(results):
        _, width, height = frames_with_meta[frame_index]
        frame_area = float(width * height)
        valid: List[Tuple[float, float, float, float]] = []

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            all_bboxes.append(valid)
            continue

        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        for detection_index in range(len(class_ids)):
            if int(class_ids[detection_index]) != 0:
                continue
            if confidences[detection_index] < confidence_threshold:
                continue

            x1, y1, x2, y2 = xyxy[detection_index]
            bbox_area = (x2 - x1) * (y2 - y1)
            if frame_area > 0 and (bbox_area / frame_area) < min_bbox_area_ratio:
                continue

            valid.append((float(x1), float(y1), float(x2), float(y2)))

        all_bboxes.append(valid)

    return all_bboxes


__all__ = ["detect_persons_batch"]
