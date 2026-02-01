"""How2Sign dataset definition."""

from typing import List

from .base import BaseDataset
from ..registry import register_dataset


@register_dataset("how2sign")
class How2SignDataset(BaseDataset):
    name = "how2sign"

    @classmethod
    def pipeline_steps(cls, mode: str) -> List[str]:
        if mode == "pose":
            return ["extract", "normalize", "webdataset"]
        elif mode == "video":
            return ["clip_video", "webdataset"]
        raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def default_config(cls) -> dict:
        return {
            "paths": {
                "root": "dataset/how2sign",
                "videos": "dataset/how2sign/videos",
                "manifest": "dataset/how2sign/how2sign_realigned_val.csv",
            },
            "extractor": {
                "keypoint_indices": [
                    5, 6, 7, 8, 11, 12,
                    23, 25, 27, 29, 31, 33, 35, 37, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                    52, 54, 56, 58,
                    71, 73, 75, 77, 79, 81, 83, 84, 85, 86,
                    87, 88, 89, 90,
                ] + list(range(91, 133)),
            },
        }
