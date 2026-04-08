"""video2compression processor: video → cropped & compressed video (.mp4).

Compresses videos by detecting all persons across the entire video and
cropping to the union bounding box that covers every detection in every
frame.  Unlike video2crop, this processor:

- Processes the whole video (no START/END time segmentation)
- Never skips multi-person videos
- Unions ALL detections (not just the largest per frame)
"""

import gc
import logging
import os
from pathlib import Path

import cv2

from .base import BaseProcessor
from .detection import create_detector, union_bbox_tuples
from .video.ffmpeg import FfmpegSamplingParams, ffmpeg_pipe_frames, clip_and_crop
from ..registry import register_processor
from ..utils.manifest import resolve_video_path

logger = logging.getLogger(__name__)


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


@register_processor("video2compression")
class Video2CompressionProcessor(BaseProcessor):
    """High-level processor: video → cropped & compressed video (.mp4).

    Detects all persons across the entire video and crops to the union
    bounding box that includes every person in every frame.

    Orchestrates:
    - video/ffmpeg_pipe for frame decoding (pass 1)
    - detection/ backends for person detection
    - video/clip_and_crop for final output (pass 2, same ffmpeg params + crop)
    """

    name = "video2compression"

    def run(self, context):
        cfg = self.config.processing
        output_dir = context.output_dir / "compressed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create building blocks
        detector = create_detector(cfg.detection, cfg.detection_config)
        ffmpeg_params = FfmpegSamplingParams(
            sample_rate=None,  # preserve native FPS — no resampling
        )

        # Load manifest
        df = context.manifest_df
        if df is None:
            self.logger.warning("No manifest loaded, nothing to process.")
            context.stats["processing"] = {"total": 0}
            return context

        video_dir = str(context.videos_dir) if context.videos_dir else ""

        processed = skipped = errors = 0
        total = len(df)

        try:
            for _, row in df.iterrows():
                sample_id = row["SAMPLE_ID"]
                output_path = str(output_dir / f"{sample_id}.mp4")

                # Skip existing (unless force_all)
                if not getattr(context, 'force_all', False) and os.path.exists(output_path):
                    skipped += 1
                    continue

                try:
                    video_path = str(resolve_video_path(row, video_dir))
                    if not os.path.exists(video_path):
                        self.logger.warning("Video not found: %s", video_path)
                        errors += 1
                        continue

                    duration = _get_video_duration(video_path)
                    if duration <= 0:
                        self.logger.warning("Cannot read duration: %s", video_path)
                        errors += 1
                        continue

                    # Pass 1: decode frames for detection (whole video)
                    frames = ffmpeg_pipe_frames(
                        video_path, 0.0, duration, ffmpeg_params,
                    )

                    if not frames:
                        errors += 1
                        continue

                    # Detect persons
                    detections = detector.detect_batch(frames)

                    # Flatten ALL detections across all frames into bbox tuples
                    all_bboxes = [
                        det.bbox
                        for frame_dets in detections
                        for det in frame_dets
                    ]

                    if not all_bboxes:
                        self.logger.debug("No detections, skipping: %s", sample_id)
                        skipped += 1
                        continue

                    # Compute union bbox covering every person in every frame
                    bbox = union_bbox_tuples(all_bboxes)

                    # Pass 2: crop whole video with union bbox
                    ok = clip_and_crop(
                        video_path, 0.0, duration,
                        bbox, ffmpeg_params, cfg.video_config,
                        output_path,
                    )
                    if ok:
                        processed += 1
                    else:
                        errors += 1

                except Exception as e:
                    self.logger.error("Error processing %s: %s", sample_id, e)
                    errors += 1

        finally:
            detector.close()
            gc.collect()

        context.stats["processing"] = {
            "total": total,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }
        self.logger.info(
            "video2compression: processed=%d skipped=%d errors=%d total=%d",
            processed, skipped, errors, total,
        )
        return context
