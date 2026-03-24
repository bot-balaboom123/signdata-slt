"""Pipeline processors."""

from .base import BaseProcessor
from .video2pose import Video2PoseProcessor
from .video2crop import Video2CropProcessor

__all__ = [
    "BaseProcessor",
    "Video2PoseProcessor",
    "Video2CropProcessor",
]
