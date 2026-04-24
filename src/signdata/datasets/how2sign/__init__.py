"""How2Sign dataset package.

Re-exports legacy module symbols so existing imports from
``signdata.datasets.how2sign`` continue to work after the package split.
"""

from . import manifest, source
from .adapter import How2SignDataset
from .manifest import build
from .source import How2SignSourceConfig, get_source_config, validate

__all__ = [
    "How2SignDataset",
    "How2SignSourceConfig",
    "build",
    "get_source_config",
    "manifest",
    "source",
    "validate",
]
