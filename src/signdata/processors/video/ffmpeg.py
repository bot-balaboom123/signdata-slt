"""FFmpeg-based video processing utilities for video2crop."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ...utils.video import resolve_effective_sample_fps

logger = logging.getLogger(__name__)


@dataclass
class FfmpegSamplingParams:
    """Shared ffmpeg parameters for consistent frame decoding across passes."""
    sample_rate: Optional[float] = 0.5


def ffmpeg_pipe_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    params: FfmpegSamplingParams,
    width: int = 0,
    height: int = 0,
) -> List[np.ndarray]:
    """Decode frames from a video segment via ffmpeg pipe.

    Pass 1 of the two-pass video2crop pipeline. Returns BGR frames
    that can be fed to the detection backend.

    Args:
        video_path: Path to the video file.
        start_sec: Segment start time.
        end_sec: Segment end time.
        params: Shared ffmpeg sampling parameters.
        width: Frame width (0 = auto-detect from video).
        height: Frame height (0 = auto-detect from video).

    Returns:
        List of BGR frames as numpy arrays.
    """
    import cv2

    # Auto-detect dimensions if not provided
    source_fps = 0.0
    if width == 0 or height == 0:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
    else:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()

    duration = end_sec - start_sec
    effective_fps = resolve_effective_sample_fps(source_fps, params.sample_rate)

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
    ]

    # Apply FPS filter
    vf_filters = []
    if effective_fps is not None:
        vf_filters.append(f"fps={effective_fps}")

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ])

    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            logger.error("ffmpeg error: %s", proc.stderr.decode()[:200])
            return []

        raw = proc.stdout
        frame_size = width * height * 3
        if len(raw) < frame_size:
            return []

        n_frames = len(raw) // frame_size
        frames = []
        for i in range(n_frames):
            frame = np.frombuffer(
                raw[i * frame_size:(i + 1) * frame_size],
                dtype=np.uint8,
            ).reshape(height, width, 3).copy()
            frames.append(frame)

        return frames

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out for %s", video_path)
        return []
    except Exception as e:
        logger.error("ffmpeg pipe error: %s", e)
        return []


def clip_and_crop(
    video_path: str,
    start_sec: float,
    end_sec: float,
    bbox: Tuple[float, float, float, float],
    params: FfmpegSamplingParams,
    video_config,
    output_path: str,
) -> bool:
    """Pass 2: clip + crop a video segment using ffmpeg.

    Uses the same timing parameters as pass 1 (ffmpeg_pipe_frames)
    plus a crop filter derived from the detection bbox.

    Args:
        video_path: Source video path.
        start_sec: Segment start time.
        end_sec: Segment end time.
        bbox: (x1, y1, x2, y2) crop region in pixels.
        params: Same FfmpegSamplingParams used in pass 1.
        video_config: VideoProcessingConfig with codec, padding, resize.
        output_path: Output file path.

    Returns:
        True if successful.
    """
    import cv2
    from ..detection.validation import apply_bbox_padding

    duration = end_sec - start_sec

    # Get frame dimensions for padding calculation
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    # Apply padding
    padding = getattr(video_config, "padding", 0.0)
    x1, y1, x2, y2 = apply_bbox_padding(bbox, padding, frame_w, frame_h)
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        return False

    effective_fps = resolve_effective_sample_fps(source_fps, params.sample_rate)

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
    ]

    vf_filters = []
    if effective_fps is not None:
        vf_filters.append(f"fps={effective_fps}")
    vf_filters.append(f"crop={crop_w}:{crop_h}:{x1}:{y1}")

    resize = getattr(video_config, "resize", None)
    if resize:
        vf_filters.append(f"scale={resize[0]}:{resize[1]}")

    cmd.extend(["-vf", ",".join(vf_filters)])

    codec = getattr(video_config, "codec", "libx264")
    cmd.extend(["-c:v", codec, "-preset", "medium", "-crf", "15", "-an"])
    cmd.extend(["-v", "error"])
    cmd.append(output_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            logger.error("ffmpeg crop error: %s", proc.stderr.decode()[:200])
            return False
        return True
    except Exception as e:
        logger.error("ffmpeg crop error: %s", e)
        return False
