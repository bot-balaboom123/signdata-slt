"""FFmpeg-based video processing utilities for video2crop."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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
    try:
        frames: List[np.ndarray] = []
        for batch in iter_ffmpeg_frame_batches(
            video_path, start_sec, end_sec, params, batch_size=64,
            width=width, height=height,
        ):
            frames.extend(batch)
        return frames
    except Exception as e:
        logger.error("ffmpeg pipe error: %s", e)
        return []


def iter_ffmpeg_frame_batches(
    video_path: str,
    start_sec: float,
    end_sec: float,
    params: FfmpegSamplingParams,
    batch_size: int,
    width: int = 0,
    height: int = 0,
) -> Iterator[List[np.ndarray]]:
    """Stream decoded frames from ffmpeg in bounded-size batches.

    Unlike :func:`ffmpeg_pipe_frames`, this function does not buffer the full
    rawvideo stream or the full decoded frame list in memory.
    """
    import cv2

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    source_fps = 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    try:
        if width == 0 or height == 0:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        return

    duration = end_sec - start_sec
    effective_fps = resolve_effective_sample_fps(source_fps, params.sample_rate)

    # -hide_banner + -nostats prevent ffmpeg from filling the stderr PIPE
    # buffer and deadlocking the streaming read loop below.
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-nostats",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
    ]

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

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout pipe was not created")

        frame_size = width * height * 3
        batch: List[np.ndarray] = []

        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                logger.warning(
                    "ffmpeg produced a partial frame for %s; dropping trailing bytes",
                    video_path,
                )
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                height, width, 3,
            ).copy()
            batch.append(frame)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        returncode = proc.wait()
        if returncode != 0:
            message = stderr.decode(errors="replace")[:200]
            raise RuntimeError(f"ffmpeg error: {message}")

    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


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
