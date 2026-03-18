"""Pipeline step implementations."""

import importlib
import pkgutil

from .base import BaseProcessor

# Auto-discover and import all processor sub-packages/modules to trigger
# @register_processor decorators.
for _, _module_name, _ in pkgutil.iter_modules(__path__):
    if _module_name != "base":
        importlib.import_module(f".{_module_name}", __package__)

from .common import ExtractProcessor, NormalizeProcessor, ClipVideoProcessor, WebDatasetProcessor
from .youtube_asl import DownloadProcessor, ManifestProcessor

__all__ = [
    "BaseProcessor",
    "DownloadProcessor",
    "ManifestProcessor",
    "ExtractProcessor",
    "NormalizeProcessor",
    "ClipVideoProcessor",
    "WebDatasetProcessor",
]
