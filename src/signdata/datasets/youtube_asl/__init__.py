"""YouTube-ASL dataset package.

Re-exports legacy module symbols so existing imports from
``signdata.datasets.youtube_asl`` continue to work after the package split.
"""

from . import manifest, source
from .adapter import YouTubeASLDataset
from .manifest import _process_segments, _save_segments, build
from .source import (
    DEFAULT_DOWNLOAD_FORMAT,
    DEFAULT_TRANSCRIPT_LANGUAGES,
    YouTubeASLSourceConfig,
    _build_transcript_client,
    _build_transcript_proxies,
    _download_transcripts,
    _download_videos,
    _fetch_transcript,
    _get_existing_ids,
    _load_video_ids,
    _normalize_transcript_payload,
    download,
    get_source_config,
    time,
)

__all__ = [
    "DEFAULT_DOWNLOAD_FORMAT",
    "DEFAULT_TRANSCRIPT_LANGUAGES",
    "YouTubeASLDataset",
    "YouTubeASLSourceConfig",
    "_build_transcript_client",
    "_build_transcript_proxies",
    "_download_transcripts",
    "_download_videos",
    "_fetch_transcript",
    "_get_existing_ids",
    "_load_video_ids",
    "_normalize_transcript_payload",
    "_process_segments",
    "_save_segments",
    "build",
    "download",
    "get_source_config",
    "manifest",
    "source",
    "time",
]
