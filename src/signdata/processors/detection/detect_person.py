"""Person detection processor: detect the signer in each video segment.

Uses YOLOv8-nano to detect persons across sampled frames, then writes
bounding box information back into the manifest CSV.

Manifest columns added:
    BBOX_X1, BBOX_Y1, BBOX_X2, BBOX_Y2  -- union bbox in pixels (float)
    PERSON_DETECTED                       -- bool, False = fallback to full frame
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..base import BaseProcessor
from .yolo import YOLO, detect_persons_batch as _detect_persons_batch
from .sampling import _sample_frames, _sample_frames_skip, _sample_frames_uniform
from .validation import union_bbox_tuples as _union_bboxes
from ...registry import register_processor
from ...utils.manifest import read_manifest, get_timing_columns, find_video_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

@register_processor("detect_person")
class DetectPersonProcessor(BaseProcessor):
    """Detect the signer across video segments.

    Reads VIDEO_ID / SAMPLE_ID / timestamps from the manifest,
    samples frames from the original video, runs YOLOv8-nano person
    detection, unions bboxes across sampled frames, and writes results
    back to the manifest CSV.

    New manifest columns:
        BBOX_X1, BBOX_Y1, BBOX_X2, BBOX_Y2  (float, pixels)
        PERSON_DETECTED                       (bool)
    """

    name = "detect_person"

    def run(self, context):
        cfg = self.config
        loc_cfg = cfg.detect_person
        manifest_path = str(context.manifest_path)
        video_dir = str(context.video_dir)

        # ----------------------------------------------------------------
        # Compute stage manifest path — detect_person writes a derived
        # manifest rather than mutating the base manifest in-place.
        # The runner sets stage_output_dir; fall back to computing it
        # when running outside the runner (e.g. smoke tests).
        # ----------------------------------------------------------------
        if context.stage_output_dir:
            stage_manifest_dir = context.stage_output_dir
        else:
            root = Path(cfg.paths.root)
            stage_manifest_dir = root / "detect_person" / cfg.run_name
        stage_manifest_dir.mkdir(parents=True, exist_ok=True)
        stage_manifest_path = str(stage_manifest_dir / "manifest.csv")

        # ----------------------------------------------------------------
        # Load manifest
        # ----------------------------------------------------------------
        data = read_manifest(manifest_path, normalize_columns=True)
        start_col, end_col = get_timing_columns(data)

        # If columns already exist from a previous run, we skip rows that
        # were already processed (PERSON_DETECTED is not NaN).
        already_done_col = "PERSON_DETECTED"
        if already_done_col not in data.columns:
            data["BBOX_X1"] = np.nan
            data["BBOX_Y1"] = np.nan
            data["BBOX_X2"] = np.nan
            data["BBOX_Y2"] = np.nan
            # Use object dtype so we can store True/False/NaN without warning
            data["PERSON_DETECTED"] = pd.array([pd.NA] * len(data), dtype="object")

        # Only process rows not yet done
        pending_mask = data["PERSON_DETECTED"].isna()
        pending = data[pending_mask].copy()

        if pending.empty:
            self.logger.info("All rows already detected, skipping.")
            context.stats["detect_person"] = {"total": 0}
            return context

        self.logger.info(
            "Detecting persons in %d segments (skipping %d already done)",
            len(pending),
            len(data) - len(pending),
        )

        # ----------------------------------------------------------------
        # Load detector model — dispatch on backend
        # ----------------------------------------------------------------
        if loc_cfg.backend == "ultralytics":
            if YOLO is None:
                raise ImportError(
                    "ultralytics is required for backend='ultralytics'. "
                    "Install with: pip install ultralytics"
                )
            self.logger.info("Loading YOLOv8 model: %s on %s", loc_cfg.model, loc_cfg.device)
            model = YOLO(loc_cfg.model)
            model.to(loc_cfg.device)
        else:
            raise ValueError(f"Unknown detector backend: {loc_cfg.backend!r}")

        # ----------------------------------------------------------------
        # Process each segment
        # ----------------------------------------------------------------
        detected = fallback = errors = 0

        # We'll accumulate results as a dict keyed by DataFrame index
        results_map = {}
        _CHECKPOINT_EVERY = 50

        for idx, row in tqdm(pending.iterrows(), total=len(pending), desc="Detecting persons"):
            video_name = row["VIDEO_ID"]
            video_path = str(find_video_file(video_dir, video_name))

            if not os.path.exists(video_path):
                self.logger.warning("Video not found, using fallback: %s", video_path)
                results_map[idx] = self._fallback_row(video_path)
                fallback += 1
            else:
                try:
                    start_sec = float(row[start_col])
                    end_sec = float(row[end_col])

                    # Sample frames — frame_skip comes from processing config (Issue 1a)
                    frames_meta = _sample_frames(
                        video_path, start_sec, end_sec,
                        strategy=loc_cfg.sample_strategy,
                        frame_skip=cfg.processing.frame_skip,
                        uniform_frames=loc_cfg.uniform_frames,
                        max_frames=loc_cfg.max_frames,
                    )

                    if not frames_meta:
                        self.logger.warning(
                            "Could not sample frames for %s, using fallback.",
                            row["SAMPLE_ID"],
                        )
                        results_map[idx] = self._fallback_row(video_path)
                        fallback += 1
                    else:
                        # Batch detection across all sampled frames
                        per_frame_bboxes = _detect_persons_batch(
                            model,
                            frames_meta,
                            loc_cfg.confidence_threshold,
                            loc_cfg.min_bbox_area,
                        )

                        # Collect all valid bboxes across frames
                        all_valid: List[Tuple[float, float, float, float]] = []
                        for frame_bboxes in per_frame_bboxes:
                            # Pick the largest-area bbox per frame (most likely the signer)
                            if frame_bboxes:
                                best = max(
                                    frame_bboxes,
                                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                                )
                                all_valid.append(best)

                        if all_valid:
                            union = _union_bboxes(all_valid)
                            results_map[idx] = {
                                "BBOX_X1": union[0],
                                "BBOX_Y1": union[1],
                                "BBOX_X2": union[2],
                                "BBOX_Y2": union[3],
                                "PERSON_DETECTED": True,
                            }
                            detected += 1
                        else:
                            # No person found in any sampled frame → fallback to full frame
                            _, w, h = frames_meta[0]
                            results_map[idx] = {
                                "BBOX_X1": 0.0,
                                "BBOX_Y1": 0.0,
                                "BBOX_X2": float(w),
                                "BBOX_Y2": float(h),
                                "PERSON_DETECTED": False,
                            }
                            fallback += 1

                except Exception as e:
                    self.logger.error(
                        "Error processing %s: %s", row.get("SAMPLE_ID", "?"), e
                    )
                    results_map[idx] = self._fallback_row(video_path)
                    errors += 1

            # Periodic checkpoint — flush results so a crash doesn't lose all progress
            if len(results_map) % _CHECKPOINT_EVERY == 0:
                for i, vals in results_map.items():
                    for col, val in vals.items():
                        data.at[i, col] = val
                data.to_csv(stage_manifest_path, sep="\t", index=False)

        # ----------------------------------------------------------------
        # Final write — flush any remaining results not caught by checkpoint
        # ----------------------------------------------------------------
        for idx, vals in results_map.items():
            for col, val in vals.items():
                data.at[idx, col] = val

        data.to_csv(stage_manifest_path, sep="\t", index=False)
        self.logger.info(
            "Stage manifest written: %s  detected=%d  fallback=%d  errors=%d",
            stage_manifest_path, detected, fallback, errors,
        )

        # Update context so downstream stages read the augmented manifest
        context.manifest_df = data

        context.stats["detect_person"] = {
            "total": len(pending),
            "detected": detected,
            "fallback": fallback,
            "errors": errors,
        }
        return context

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _fallback_row(video_path: str) -> dict:
        """Return a full-frame fallback entry when detection is impossible."""
        w, h = 0.0, 0.0
        if os.path.exists(video_path):
            video_capture = cv2.VideoCapture(video_path)
            if video_capture.isOpened():
                w = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                video_capture.release()
        return {
            "BBOX_X1": 0.0,
            "BBOX_Y1": 0.0,
            "BBOX_X2": float(w),
            "BBOX_Y2": float(h),
            "PERSON_DETECTED": False,
        }
