"""Normalize post-processor: visibility masking + landmark normalization."""

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .base import BasePostProcessor
from ..pose.presets import resolve_keypoint_indices
from ..registry import register_post_processor


# Reuse core math from processors/pose/normalize.py
from ..processors.pose.normalize import (
    _load_clip,
    _apply_keypoint_reduction,
    _apply_visibility_mask,
    _normalize_clip_xyz,
)


def _process_single_file(args) -> Tuple[str, bool, str]:
    """Process a single landmark file through the normalization pipeline."""
    input_path, output_path, normalize_config = args
    filename = os.path.basename(input_path)

    try:
        if normalize_config.get("skip_existing") and os.path.exists(output_path):
            return filename, True, "skipped (exists)"

        clip_raw = _load_clip(input_path)

        # Keypoint reduction
        if normalize_config["select_keypoints"]:
            indices = resolve_keypoint_indices(
                normalize_config.get("keypoint_preset"),
                normalize_config.get("keypoint_indices"),
            )
            if indices is not None:
                clip_raw = _apply_keypoint_reduction(clip_raw, indices)

        # Visibility masking
        xyz_masked = _apply_visibility_mask(
            clip_raw,
            normalize_config["mask_empty_frames"],
            normalize_config["mask_low_confidence"],
            normalize_config["visibility_threshold"],
            normalize_config["missing_value"],
        )

        # Normalization
        xyz_norm = _normalize_clip_xyz(
            xyz_masked, normalize_config["mode"], normalize_config["missing_value"]
        )

        # Optionally drop z
        if normalize_config["remove_z"]:
            xyz_norm = xyz_norm[..., :2]

        # Flatten
        T, K, C_out = xyz_norm.shape
        flattened = xyz_norm.reshape(T, K * C_out).astype(np.float32)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, flattened)

        return filename, True, ""
    except Exception as e:
        return filename, False, str(e)


@register_post_processor("normalize")
class NormalizePostProcessor(BasePostProcessor):
    """Normalize pose landmarks as a post-processing step.

    Reads from context.output_dir/raw, writes to context.output_dir/normalized.
    """

    name = "normalize"

    def run(self, context):
        cfg = self.config
        norm_cfg = cfg.post_processing.normalize

        if norm_cfg is None:
            self.logger.warning("No normalize config provided, skipping.")
            context.stats["post_processing.normalize"] = {"total": 0}
            return context

        input_dir = str(context.output_dir / "raw")
        output_dir = str(context.output_dir / "normalized")
        os.makedirs(output_dir, exist_ok=True)

        pattern = os.path.join(input_dir, "**", "*.npy")
        npy_files = glob.glob(pattern, recursive=True)

        if not npy_files:
            self.logger.warning("No .npy files found in %s", input_dir)
            context.stats["post_processing.normalize"] = {"total": 0}
            return context

        normalize_config = {
            "mode": norm_cfg.mode,
            "remove_z": norm_cfg.remove_z,
            "select_keypoints": norm_cfg.select_keypoints,
            "keypoint_preset": norm_cfg.keypoint_preset,
            "keypoint_indices": norm_cfg.keypoint_indices,
            "mask_empty_frames": norm_cfg.mask_empty_frames,
            "mask_low_confidence": norm_cfg.mask_low_confidence,
            "visibility_threshold": norm_cfg.visibility_threshold,
            "missing_value": norm_cfg.missing_value,
            "skip_existing": True,
        }

        tasks = []
        for inp in npy_files:
            rel = os.path.relpath(inp, input_dir)
            out = os.path.join(output_dir, rel)
            tasks.append((inp, out, normalize_config))

        self.logger.info("Normalizing %d files from %s", len(tasks), input_dir)

        max_workers = cfg.processing.max_workers
        success = skip = errors = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_single_file, t): t for t in tasks
            }
            with tqdm(total=len(tasks), desc="Normalizing") as pbar:
                for future in as_completed(futures):
                    fname, ok, msg = future.result()
                    if ok:
                        if msg == "skipped (exists)":
                            skip += 1
                        else:
                            success += 1
                    else:
                        errors += 1
                        self.logger.error("Failed: %s - %s", fname, msg)
                    pbar.update(1)

        context.stats["post_processing.normalize"] = {
            "total": len(tasks),
            "success": success,
            "skipped": skip,
            "errors": errors,
        }
        return context
