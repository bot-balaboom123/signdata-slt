"""Frame sampling helpers for detection-oriented processors."""

from typing import List, Tuple

import cv2
import numpy as np


def _sample_frames_uniform(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n: int,
) -> List[Tuple[np.ndarray, int, int]]:
    """Uniformly sample exactly n frames from [start_sec, end_sec]."""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return []

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        video_capture.release()
        return []

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_frames = max(end_frame - start_frame, 1)

    if n == 1:
        indices = [start_frame + total_frames // 2]
    else:
        step = total_frames / (n - 1)
        indices = [int(start_frame + i * step) for i in range(n)]
        indices = [min(idx, end_frame) for idx in indices]

    frames = []
    for idx in indices:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = video_capture.read()
        if ok and frame is not None:
            frames.append((frame, width, height))

    video_capture.release()
    return frames


def _sample_frames_skip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    frame_skip: int,
    max_frames: int,
) -> List[Tuple[np.ndarray, int, int]]:
    """Sample frames by skipping every frame_skip frames, up to max_frames."""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return []

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        video_capture.release()
        return []

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    current = start_frame
    while current <= end_frame and len(frames) < max_frames:
        ok, frame = video_capture.read()
        if not ok:
            break
        if frame is not None:
            frames.append((frame, width, height))
        current += frame_skip + 1
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current)

    video_capture.release()
    return frames


def _sample_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    strategy: str,
    frame_skip: int,
    uniform_frames: int,
    max_frames: int,
) -> List[Tuple[np.ndarray, int, int]]:
    """Dispatch to the appropriate sampling strategy."""
    if strategy == "skip_frame":
        return _sample_frames_skip(
            video_path,
            start_sec,
            end_sec,
            frame_skip=frame_skip,
            max_frames=max_frames,
        )
    return _sample_frames_uniform(
        video_path,
        start_sec,
        end_sec,
        n=uniform_frames,
    )


__all__ = [
    "_sample_frames",
    "_sample_frames_skip",
    "_sample_frames_uniform",
]
