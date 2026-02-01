"""YouTube-ASL dataset definition."""

from typing import List

from .base import BaseDataset
from ..registry import register_dataset


@register_dataset("youtube_asl")
class YouTubeASLDataset(BaseDataset):
    name = "youtube_asl"

    @classmethod
    def pipeline_steps(cls, mode: str) -> List[str]:
        if mode == "pose":
            return ["download", "manifest", "extract", "normalize", "webdataset"]
        elif mode == "video":
            return ["download", "manifest", "clip_video", "webdataset"]
        raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def default_config(cls) -> dict:
        return {
            "paths": {
                "root": "dataset/youtube_asl",
                "videos": "dataset/youtube_asl/videos",
                "transcripts": "dataset/youtube_asl/transcripts",
                "manifest": "dataset/youtube_asl/manifest.csv",
            },
            "download": {
                "video_ids_file": "assets/youtube-asl_youtube_asl_video_ids.txt",
                "languages": [
                    "en", "ase", "en-US", "en-CA", "en-GB", "en-AU",
                ],
                "format": (
                    "worstvideo[height>=720][fps>=24]"
                    "/bestvideo[height>=480][height<720][fps>=24][fps<=60]"
                    "/bestvideo[height>=480][height<=1080][fps>=14]"
                ),
                "rate_limit": "5M",
            },
            "manifest": {
                "max_text_length": 300,
                "min_duration": 0.2,
                "max_duration": 60.0,
            },
            "extractor": {
                "pose_idx": [11, 12, 13, 14, 23, 24],
                "face_idx": [
                    0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64,
                    81, 82, 93, 133, 151, 152, 159, 172, 178, 181, 263,
                    269, 276, 282, 285, 291, 294, 311, 323, 362, 386,
                    397, 468, 473,
                ],
                "hand_idx": list(range(21)),
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
