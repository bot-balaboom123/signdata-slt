"""Pipeline step implementations."""

from .base import BaseProcessor
from .detection import DetectPersonProcessor
from .output import WebDatasetProcessor
from .pose import ExtractProcessor, NormalizeProcessor
from .video import (
    ClipVideoProcessor,
    CropVideoProcessor,
    ObfuscateProcessor,
    WindowVideoProcessor,
)

__all__ = [
    "BaseProcessor",
    "DetectPersonProcessor",
    "WindowVideoProcessor",
    "ClipVideoProcessor",
    "CropVideoProcessor",
    "ObfuscateProcessor",
    "ExtractProcessor",
    "NormalizeProcessor",
    "WebDatasetProcessor",
]
