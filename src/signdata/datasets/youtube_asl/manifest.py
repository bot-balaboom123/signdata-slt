"""YouTube-ASL manifest building."""

import csv
import json
import os
import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .._shared.availability import apply_availability_policy
from ...utils.text import normalize_text
from .source import YouTubeASLSourceConfig


def build(
    config,
    source: YouTubeASLSourceConfig,
    log: logging.Logger,
):
    """Build segmented manifest from transcript JSON files."""
    transcript_dir = config.paths.transcripts
    manifest_path = config.paths.manifest

    json_files = glob(os.path.join(transcript_dir, "*.json"))

    if not json_files:
        log.warning("No transcript files found in %s", transcript_dir)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        empty_df = pd.DataFrame(
            columns=["VIDEO_ID", "SAMPLE_ID", "START", "END", "TEXT"]
        )
        empty_df.to_csv(manifest_path, sep="\t", index=False)
        return manifest_path, empty_df, {"videos": 0, "segments": 0}

    log.info(
        "Processing %d transcript files from %s",
        len(json_files),
        transcript_dir,
    )

    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    text_opts = source.text_processing.model_dump()
    processed_count = 0
    total_segments = 0
    first_write = True

    for json_file in tqdm(json_files, desc="Building manifest"):
        video_id = os.path.splitext(os.path.basename(json_file))[0]
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)

            if not transcript_data:
                continue

            segments = _process_segments(
                transcript_data,
                video_id,
                source.max_text_length,
                source.min_duration,
                source.max_duration,
                text_opts,
            )

            if segments:
                _save_segments(segments, manifest_path, append=not first_write)
                first_write = False
                processed_count += 1
                total_segments += len(segments)

        except Exception as e:
            log.error("Error processing %s: %s", video_id, e)

    if os.path.exists(manifest_path):
        from ...utils.manifest import read_manifest

        df = read_manifest(manifest_path, normalize_columns=True)

        video_dir = config.paths.videos
        if video_dir and Path(video_dir).is_dir():
            df = apply_availability_policy(
                df, video_dir, source.availability_policy,
            )
            df.to_csv(manifest_path, sep="\t", index=False)
    else:
        df = None

    return manifest_path, df, {
        "videos": processed_count,
        "segments": total_segments,
    }


def _process_segments(
    transcripts: List[Dict],
    video_id: str,
    max_text_length: int,
    min_duration: float,
    max_duration: float,
    text_options: Optional[Dict] = None,
) -> List[Dict]:
    processed = []
    idx = 0

    valid = [
        t for t in transcripts
        if "text" in t and "start" in t and "duration" in t
    ]

    text_kw = text_options or {}

    for entry in valid:
        text = normalize_text(entry["text"], **text_kw)
        dur = entry["duration"]

        if (
            text
            and len(text) <= max_text_length
            and min_duration <= dur <= max_duration
        ):
            processed.append({
                "VIDEO_ID": video_id,
                "SAMPLE_ID": f"{video_id}-{idx:03d}",
                "START": entry["start"],
                "END": entry["start"] + dur,
                "TEXT": text,
            })
            idx += 1

    return processed


def _save_segments(
    segments: List[Dict],
    csv_path: str,
    append: bool = False,
) -> None:
    df = pd.DataFrame(segments)
    mode = "a" if append else "w"
    header = not append
    df.to_csv(
        csv_path,
        sep="\t",
        mode=mode,
        header=header,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )
