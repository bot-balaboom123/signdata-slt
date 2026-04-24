"""OpenASL dataset package.

Re-exports legacy module symbols so existing imports from
``signdata.datasets.openasl`` continue to work after the package split.
"""

from . import manifest, source
from .adapter import OpenASLDataset
from .manifest import _merge_bboxes, build
from .source import (
    OpenASLSourceConfig,
    _download_videos,
    download,
    get_source_config,
)

__all__ = [
    "OpenASLDataset",
    "OpenASLSourceConfig",
    "_download_videos",
    "_merge_bboxes",
    "build",
    "download",
    "get_source_config",
    "manifest",
    "source",
]
