"""Frame reading with sampling via OpenCV."""

from typing import List, Optional

import cv2
import numpy as np

from .base import BaseSampler


def read_sampled_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sampler: BaseSampler,
    source_fps: Optional[float] = None,
) -> List[np.ndarray]:
    """Read and sample frames from a video segment using OpenCV.

    Args:
        video_path: Path to the video file.
        start_sec: Segment start time in seconds.
        end_sec: Segment end time in seconds.
        sampler: A BaseSampler instance that decides which frames to keep.
        source_fps: Override source FPS. If None, read from video metadata.

    Returns:
        List of BGR frames (np.ndarray).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = source_fps or cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        cap.release()
        return []

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sampler.reset()
    frames = []
    current = start_frame

    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if sampler.take():
            frames.append(frame)
        current += 1

    cap.release()
    return frames
