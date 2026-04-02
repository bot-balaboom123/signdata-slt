"""Dataset ingestion helpers (download + manifest-build stages).

Used exclusively by dataset adapters (``source.py`` and ``manifest.py``).
Must NOT be imported by processors, pipeline runner, or output modules.

For pipeline-wide utilities (FPS sampling, manifest I/O, video validation),
use ``signdata.utils``.

Submodules
----------
availability    -- AvailabilityPolicy, existence checks, policy enforcement
classmap        -- TSV class-map loading and joining
media           -- video duration/FPS probing; frame-sequence materialisation
text            -- dataset-ingestion text normalization
youtube         -- yt-dlp download wrapper
"""

from pathlib import Path

from .availability import (
    AvailabilityPolicy,
    apply_availability_policy,
    apply_availability_policy_paths,
    get_existing_video_ids,
    write_acquire_report,
)
from .classmap import join_class_map, load_class_map
from .media import get_video_duration, get_video_fps, materialize_frames_to_video
from .text import TextProcessingConfig, normalize_text
from .youtube import DownloadResult, download_youtube_videos


def resolve_dir(primary: str, fallback: str = "") -> Path:
    """Return a Path from *primary*, falling back to *fallback*.

    Useful for resolving ``release_dir`` with a ``paths.videos`` fallback::

        release_dir = resolve_dir(source.release_dir, config.paths.videos or "")
    """
    return Path(primary or fallback or "")

__all__ = [
    "AvailabilityPolicy",
    "apply_availability_policy",
    "apply_availability_policy_paths",
    "get_existing_video_ids",
    "write_acquire_report",
    "join_class_map",
    "load_class_map",
    "get_video_duration",
    "get_video_fps",
    "materialize_frames_to_video",
    "resolve_dir",
    "TextProcessingConfig",
    "normalize_text",
    "DownloadResult",
    "download_youtube_videos",
]
