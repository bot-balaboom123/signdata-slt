"""Named keypoint reduction presets for pose outputs.

Preset data lives in per-preset modules. This package assembles the
registry and exposes helpers used by normalization and config validation.
"""

from typing import Dict, List, Optional

from .mediapipe_543_to_83 import PRESET as MEDIAPIPE_543_TO_83
from .mediapipe_553_to_85 import PRESET as MEDIAPIPE_553_TO_85
from .mmpose_133_to_85 import PRESET as MMPOSE_133_TO_85

KEYPOINT_PRESETS: Dict[str, dict] = {
    "mediapipe_553_to_85": MEDIAPIPE_553_TO_85,
    "mediapipe_543_to_83": MEDIAPIPE_543_TO_83,
    "mmpose_133_to_85": MMPOSE_133_TO_85,
}


def resolve_keypoint_indices(
    preset_name: Optional[str] = None,
    manual_indices: Optional[List[int]] = None,
) -> Optional[List[int]]:
    """Resolve keypoint indices from a preset name or manual list."""
    if preset_name is not None:
        if preset_name not in KEYPOINT_PRESETS:
            raise ValueError(
                f"Unknown keypoint preset '{preset_name}'. "
                f"Available: {sorted(KEYPOINT_PRESETS.keys())}"
            )
        return list(KEYPOINT_PRESETS[preset_name]["indices"])

    if manual_indices is not None:
        return list(manual_indices)

    return None


def list_presets() -> Dict[str, str]:
    """Return a mapping of preset names to their descriptions."""
    return {
        name: info["description"]
        for name, info in KEYPOINT_PRESETS.items()
    }


__all__ = [
    "KEYPOINT_PRESETS",
    "list_presets",
    "resolve_keypoint_indices",
]
