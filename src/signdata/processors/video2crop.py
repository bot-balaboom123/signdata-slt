"""video2crop processor: video → cropped video (.mp4)."""

import gc
import logging
import os
from pathlib import Path

from .base import BaseProcessor
from .detection import create_detector, single_person_check, union_bboxes
from .video.processing import FfmpegSamplingParams, ffmpeg_pipe_frames, clip_and_crop
from ..registry import register_processor
from ..utils.manifest import get_timing_columns, resolve_video_path

logger = logging.getLogger(__name__)


@register_processor("video2crop")
class Video2CropProcessor(BaseProcessor):
    """High-level processor: video → cropped video (.mp4).

    Uses ffmpeg as the single frame source for both detection and output,
    ensuring frame-level consistency (no OpenCV/ffmpeg mismatch).

    Orchestrates:
    - video/ffmpeg_pipe for frame decoding (pass 1)
    - detection/ backends for person detection
    - video/clip_and_crop for final output (pass 2, same ffmpeg params + crop)
    """

    name = "video2crop"

    def run(self, context):
        cfg = self.config.processing
        output_dir = context.output_dir / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create building blocks
        detector = create_detector(cfg.detection, cfg.detection_config)
        ffmpeg_params = FfmpegSamplingParams(
            target_fps=cfg.target_fps,
            frame_skip=cfg.frame_skip,
        )

        # Load manifest
        df = context.manifest_df
        if df is None:
            self.logger.warning("No manifest loaded, nothing to process.")
            context.stats["processing"] = {"total": 0}
            return context

        start_col, end_col = get_timing_columns(df)
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

                    start_sec = float(row[start_col])
                    end_sec = float(row[end_col])

                    # Pass 1: decode frames for detection
                    frames = ffmpeg_pipe_frames(
                        video_path, start_sec, end_sec, ffmpeg_params,
                    )

                    if not frames:
                        errors += 1
                        continue

                    # Subsample frames for detection (frame_skip reduces compute)
                    detect_frames = frames[::cfg.frame_skip] if cfg.frame_skip > 1 else frames

                    # Detect persons
                    detections = detector.detect_batch(detect_frames)

                    # Validate single person
                    if not single_person_check(detections):
                        self.logger.debug("Multi-person detected, skipping: %s", sample_id)
                        skipped += 1
                        continue

                    # Compute union bbox across all frames
                    bbox = union_bboxes(detections)
                    if bbox is None:
                        self.logger.debug("No detections, skipping: %s", sample_id)
                        skipped += 1
                        continue

                    # Pass 2: clip + crop with same params
                    ok = clip_and_crop(
                        video_path, start_sec, end_sec,
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
            "video2crop: processed=%d skipped=%d errors=%d total=%d",
            processed, skipped, errors, total,
        )
        return context
