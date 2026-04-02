"""Pipeline utilities shared across processors, output, and runner.

Dataset-ingestion helpers (download, class maps, text normalization,
availability enforcement) live in ``signdata.datasets._ingestion``.
"""

from .video import (
    FPSSampler,
    validate_video_file,
    resolve_effective_sample_fps,
    get_video_filenames,
    get_filenames,
)
from .manifest import (
    read_manifest,
    validate_manifest,
    has_timing,
    find_video_file,
    resolve_video_path,
    get_timing_columns,
    REQUIRED_COLUMNS,
    TIMING_COLUMNS,
    LABEL_COLUMNS,
    SPATIAL_COLUMNS,
    METADATA_COLUMNS,
    ALL_KNOWN_COLUMNS,
)
from .availability import (
    filter_available,
    AvailabilityPolicy,
)

__all__ = [
    "FPSSampler",
    "validate_video_file",
    "resolve_effective_sample_fps",
    "get_video_filenames",
    "get_filenames",
    "read_manifest",
    "validate_manifest",
    "has_timing",
    "find_video_file",
    "resolve_video_path",
    "get_timing_columns",
    "REQUIRED_COLUMNS",
    "TIMING_COLUMNS",
    "LABEL_COLUMNS",
    "SPATIAL_COLUMNS",
    "METADATA_COLUMNS",
    "ALL_KNOWN_COLUMNS",
    "filter_available",
    "AvailabilityPolicy",
]
