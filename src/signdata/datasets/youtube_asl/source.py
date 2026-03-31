"""YouTube-ASL source config and acquisition."""

import json
import logging
import os
import time
from glob import glob
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel
from tqdm import tqdm

from .._shared.availability import (
    AvailabilityPolicy,
    get_existing_video_ids,
    write_acquire_report,
)
from .._shared.youtube import download_youtube_videos
from ...utils.text import TextProcessingConfig

DEFAULT_TRANSCRIPT_LANGUAGES = [
    "en",
    "ase",
    "en-US",
    "en-CA",
    "en-GB",
    "en-AU",
    "en-NZ",
    "en-IN",
    "en-ZA",
    "en-IE",
    "en-SG",
    "en-PH",
    "en-NG",
    "en-PK",
    "en-JM",
]

DEFAULT_DOWNLOAD_FORMAT = (
    "worstvideo[height>=720][fps>=24]+worstaudio"
    "/bestvideo[height>=480][height<720][fps>=24][fps<=60]+worstaudio"
    "/bestvideo[height>=480][height<=1080][fps>=14]+worstaudio"
    "/best"
)


class YouTubeASLSourceConfig(BaseModel):
    """Typed config for YouTube-ASL adapter."""

    video_ids_file: str = ""
    languages: List[str] = DEFAULT_TRANSCRIPT_LANGUAGES.copy()
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_format: str = DEFAULT_DOWNLOAD_FORMAT
    rate_limit: str = "5M"
    concurrent_fragments: int = 5
    transcript_proxy_http: Optional[str] = None
    transcript_proxy_https: Optional[str] = None
    stop_on_transcript_block: bool = True
    max_text_length: int = 300
    min_duration: float = 0.2
    max_duration: float = 60.0
    text_processing: TextProcessingConfig = TextProcessingConfig()


def get_source_config(config) -> YouTubeASLSourceConfig:
    return YouTubeASLSourceConfig(**config.dataset.source)


def download(
    source: YouTubeASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Download YouTube videos and transcripts."""
    video_dir = config.paths.videos
    transcript_dir = config.paths.transcripts

    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    log.info("Starting transcript download...")
    transcript_stats = _download_transcripts(
        source.video_ids_file,
        transcript_dir,
        source,
        log,
    )

    log.info("Starting video download...")
    video_result = _download_videos(
        source.video_ids_file,
        video_dir,
        source,
        log,
    )
    missing = video_result.pop("missing")
    video_stats = video_result

    report_dir = os.path.join(config.paths.root, "acquire_report")
    write_acquire_report(report_dir, video_stats, missing)

    if source.availability_policy == "fail_fast" and video_stats["errors"] > 0:
        raise RuntimeError(
            f"{video_stats['errors']} download(s) failed with "
            f"availability_policy='fail_fast'. "
            f"See {report_dir}/download_report.json for details."
        )

    return {
        "transcripts": transcript_stats,
        "videos": video_stats,
    }


def _get_existing_ids(directory: str, ext: str) -> Set[str]:
    """Return set of IDs from files with the specified extension."""
    files = glob(os.path.join(directory, f"*.{ext}"))
    return {os.path.splitext(os.path.basename(f))[0] for f in files}


def _load_video_ids(file_path: str) -> Set[str]:
    """Load video IDs from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _download_transcripts(
    video_id_file: str,
    transcript_dir: str,
    source: YouTubeASLSourceConfig,
    log: logging.Logger,
) -> Dict:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        IpBlocked,
        NoTranscriptFound,
        RequestBlocked,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    existing_ids = _get_existing_ids(transcript_dir, "json")
    all_ids = _load_video_ids(video_id_file)
    ids = sorted(all_ids - existing_ids)

    if not ids:
        log.info("All transcripts already downloaded.")
        return {
            "total": len(all_ids),
            "attempted": 0,
            "downloaded": 0,
            "errors": 0,
            "blocked": False,
        }

    sleep_time = 0.2
    error_count = 0
    downloaded = 0
    blocked = False
    proxies = _build_transcript_proxies(source)
    transcript_client = _build_transcript_client(source)

    with tqdm(ids, desc="Downloading transcripts") as pbar:
        for video_id in pbar:
            sleep_time = min(sleep_time, 2)
            time.sleep(sleep_time)
            try:
                transcript = _fetch_transcript(
                    transcript_client=transcript_client,
                    transcript_api_cls=YouTubeTranscriptApi,
                    video_id=video_id,
                    languages=source.languages,
                    proxies=proxies,
                )
                transcript = _normalize_transcript_payload(transcript)
                path = os.path.join(transcript_dir, f"{video_id}.json")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(transcript))
                downloaded += 1
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
                log.warning("Transcript unavailable for %s: %s", video_id, e)
                error_count += 1
            except (RequestBlocked, IpBlocked) as e:
                error_count += 1
                blocked = True
                log.error("Transcript download blocked for %s: %s", video_id, e)
                if source.stop_on_transcript_block:
                    log.error(
                        "Stopping transcript download early after an IP block. "
                        "Set dataset.source.transcript_proxy_http / "
                        "dataset.source.transcript_proxy_https or use a rotating "
                        "residential proxy to continue."
                    )
                    pbar.set_postfix(errors=error_count, blocked=1)
                    break
            except Exception as e:
                sleep_time += 0.1
                log.error("Error downloading transcript for %s: %s", video_id, e)
                error_count += 1
            pbar.set_postfix(errors=error_count)

    return {
        "total": len(all_ids),
        "attempted": downloaded + error_count,
        "downloaded": downloaded,
        "errors": error_count,
        "blocked": blocked,
    }


def _build_transcript_proxies(
    source: YouTubeASLSourceConfig,
) -> Optional[Dict[str, str]]:
    if not source.transcript_proxy_http and not source.transcript_proxy_https:
        return None

    return {
        "http": source.transcript_proxy_http or source.transcript_proxy_https,
        "https": source.transcript_proxy_https or source.transcript_proxy_http,
    }


def _build_transcript_client(source: YouTubeASLSourceConfig) -> Optional[Any]:
    from youtube_transcript_api import YouTubeTranscriptApi

    proxy_config = None
    if source.transcript_proxy_http or source.transcript_proxy_https:
        from youtube_transcript_api.proxies import GenericProxyConfig

        proxy_config = GenericProxyConfig(
            http_url=source.transcript_proxy_http,
            https_url=source.transcript_proxy_https,
        )

    try:
        return YouTubeTranscriptApi(proxy_config=proxy_config)
    except TypeError:
        if proxy_config is not None:
            return None

    try:
        return YouTubeTranscriptApi()
    except TypeError:
        return None


def _fetch_transcript(
    transcript_client: Optional[Any],
    transcript_api_cls: Any,
    video_id: str,
    languages: List[str],
    proxies: Optional[Dict[str, str]] = None,
) -> Any:
    if transcript_client is not None:
        if hasattr(transcript_client, "fetch"):
            return transcript_client.fetch(video_id, languages=languages)
        if hasattr(transcript_client, "list"):
            return transcript_client.list(video_id).find_transcript(
                languages
            ).fetch()

    if hasattr(transcript_api_cls, "list_transcripts"):
        return transcript_api_cls.list_transcripts(
            video_id, proxies=proxies
        ).find_transcript(languages).fetch()

    kwargs: Dict[str, Any] = {"languages": languages}
    if proxies is not None:
        kwargs["proxies"] = proxies
    return transcript_api_cls.get_transcript(video_id, **kwargs)


def _normalize_transcript_payload(transcript: Any) -> List[Dict]:
    if hasattr(transcript, "to_raw_data"):
        transcript = transcript.to_raw_data()

    if isinstance(transcript, list):
        return transcript

    raise TypeError(
        "Unexpected transcript payload type "
        f"{type(transcript).__name__}; expected a list or object with "
        "to_raw_data()."
    )


def _download_videos(
    video_id_file: str,
    video_dir: str,
    source: YouTubeASLSourceConfig,
    log: logging.Logger,
) -> Dict:
    existing_ids = get_existing_video_ids(video_dir)
    all_ids = _load_video_ids(video_id_file)
    ids = list(all_ids - existing_ids)

    if not ids:
        log.info("All videos already downloaded.")
        return {
            "total": len(all_ids),
            "downloaded": 0,
            "errors": 0,
            "missing": [],
        }

    result = download_youtube_videos(
        ids,
        video_dir,
        download_format=source.download_format,
        rate_limit=source.rate_limit,
        concurrent_fragments=source.concurrent_fragments,
        log=log,
    )

    return {
        "total": len(all_ids),
        "downloaded": result["downloaded"],
        "errors": result["errors"],
        "missing": result["missing"],
    }
