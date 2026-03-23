"""Backward-compatible re-export of pose presets."""

from ..pose.presets import KEYPOINT_PRESETS, list_presets, resolve_keypoint_indices

__all__ = ["KEYPOINT_PRESETS", "list_presets", "resolve_keypoint_indices"]
