"""Video processing utilities."""

import os
from typing import Optional

import cv2


def validate_video_file(video_path: str) -> bool:
    """Validate if a video file exists and can be opened by OpenCV."""
    if not os.path.exists(video_path):
        return False
    try:
        video_capture = cv2.VideoCapture(video_path)
        is_valid = video_capture.isOpened()
        video_capture.release()
        return is_valid
    except Exception:
        return False


def get_video_fps(video_path: str) -> float:
    """Return video FPS (frames per second) as float."""
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            return 0.0
        fps = video_capture.get(cv2.CAP_PROP_FPS) or 0.0
        video_capture.release()
        return float(fps)
    except Exception:
        return 0.0


def resolve_effective_sample_fps(
    src_fps: float,
    sample_rate: Optional[float],
) -> Optional[float]:
    """Resolve a user-facing sample rate to an effective FPS.

    Rules:
      - ``None`` => native FPS (no resampling)
      - ``0 < sample_rate < 1`` => keep that ratio of source frames
      - ``sample_rate >= 1`` => downsample to that absolute FPS
    """
    if sample_rate is None:
        return None

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive or null")

    if 0 < sample_rate < 1:
        if src_fps <= 0:
            return None
        return src_fps * sample_rate

    if src_fps > 0:
        return min(sample_rate, src_fps)

    return sample_rate


class FPSSampler:
    """Frame sampler using native, ratio, or absolute-FPS semantics."""

    def __init__(self, src_fps: float, sample_rate: Optional[float]):
        if sample_rate is None:
            self.mode = "native"
        elif 0 < sample_rate < 1:
            self.mode = "ratio"
        else:
            self.mode = "fps"

        effective_fps = resolve_effective_sample_fps(src_fps, sample_rate)
        self.target = src_fps if effective_fps is None else effective_fps
        self.r = 1.0 if src_fps <= 0 else (self.target / max(src_fps, 1e-6))
        self.acc = 0.0

    def take(self) -> bool:
        """Returns True if current frame should be sampled."""
        self.acc += self.r
        if self.acc >= 1.0:
            self.acc -= 1.0
            return True
        return False
