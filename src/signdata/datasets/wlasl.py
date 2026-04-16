"""WLASL dataset adapter.

WLASL expects videos to be downloaded and preprocessed with the official
start-kit scripts so that ``paths.videos`` contains one clip per
``video_id``. The adapter validates the local files and builds a canonical
manifest from ``WLASL_v0.3.json``.

Source config (parsed from ``config.dataset.source``):
    annotation_json: str          — path to the official WLASL_v0.3.json
    split: str                    — all | train | val | test
    availability_policy: str      — fail_fast | drop_unavailable | mark_unavailable
"""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

import cv2
import pandas as pd
from pydantic import BaseModel

from .base import DatasetAdapter
from ..registry import register_dataset
from ..utils.availability import (
    AvailabilityPolicy,
    apply_availability_policy,
    get_existing_video_ids,
)
from ..utils.manifest import find_video_file


class WLASLSourceConfig(BaseModel):
    """Typed config for the WLASL adapter."""

    annotation_json: str = ""
    split: Literal["all", "train", "val", "test"] = "all"
    availability_policy: AvailabilityPolicy = "drop_unavailable"


@register_dataset("wlasl")
class WLASLDataset(DatasetAdapter):
    """WLASL dataset adapter."""

    name = "wlasl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = config.dataset.source
        if not source.get("annotation_json"):
            raise ValueError(
                "wlasl requires dataset.source.annotation_json pointing to "
                "the official WLASL_v0.3.json file"
            )

    def get_source_config(self, config) -> WLASLSourceConfig:
        return WLASLSourceConfig(**config.dataset.source)

    def download(self, config, context):
        """Validate local WLASL annotation and preprocessed clip directory."""
        source = self.get_source_config(config)
        annotation_path = Path(source.annotation_json)
        video_dir = config.paths.videos

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"WLASL annotation JSON not found: {annotation_path}"
            )

        if not video_dir:
            raise ValueError(
                "paths.videos is required for WLASL. "
                "Point it to the official preprocessed videos directory."
            )

        video_path = Path(video_dir)
        if not video_path.is_dir():
            raise FileNotFoundError(
                f"WLASL video directory not found: {video_dir}\n"
                "Download and preprocess WLASL with the official start-kit "
                "before running the pipeline."
            )

        context.stats["dataset.download"] = {
            "validated": True,
            "clips_found": len(get_existing_video_ids(video_dir)),
        }
        self.logger.info(
            "WLASL inputs validated: annotation=%s videos=%s",
            annotation_path,
            video_path,
        )
        return context

    def build_manifest(self, config, context):
        """Build canonical manifest from WLASL annotations."""
        source = self.get_source_config(config)
        manifest_path = config.paths.manifest
        video_dir = config.paths.videos

        if not manifest_path:
            raise ValueError(
                "paths.manifest is required for WLASL. Set it directly or via paths.root."
            )
        if not video_dir:
            raise ValueError(
                "paths.videos is required for WLASL. "
                "Point it to the official preprocessed videos directory."
            )

        annotation_path = Path(source.annotation_json)
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"WLASL annotation JSON not found: {annotation_path}"
            )
        if not Path(video_dir).is_dir():
            raise FileNotFoundError(
                f"WLASL video directory not found: {video_dir}\n"
                "Download and preprocess WLASL with the official start-kit "
                "before running the pipeline."
            )

        with open(annotation_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        if not isinstance(content, list):
            raise ValueError(
                "WLASL annotation JSON must be a list of gloss entries."
            )

        available_ids = (
            get_existing_video_ids(video_dir)
            if Path(video_dir).is_dir()
            else set()
        )
        rows: List[Dict] = []

        for class_id, entry in enumerate(content):
            gloss = str(entry.get("gloss", "")).strip()
            instances = entry.get("instances", [])

            if not isinstance(instances, list):
                raise ValueError(
                    f"WLASL gloss entry {gloss or class_id!r} has invalid 'instances'."
                )

            for inst in instances:
                inst_split = str(inst.get("split", "")).strip()
                if source.split != "all" and inst_split != source.split:
                    continue

                video_id = str(inst.get("video_id", "")).strip()
                if not video_id:
                    continue

                fps = self._as_float(inst.get("fps"))
                frame_start = self._as_int(inst.get("frame_start"))
                frame_end = self._as_int(inst.get("frame_end"))
                duration = self._resolve_duration(
                    video_dir=video_dir,
                    video_id=video_id,
                    fps=fps,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    available_ids=available_ids,
                )

                row: Dict = {
                    "SAMPLE_ID": video_id,
                    "VIDEO_ID": video_id,
                    "START": 0.0,
                    "END": duration,
                    "GLOSS": gloss,
                    "CLASS_ID": class_id,
                    "SPLIT": inst_split,
                    "SIGNER_ID": self._as_int(inst.get("signer_id")),
                    "FPS": fps,
                    "VARIATION_ID": self._as_int(inst.get("variation_id")),
                    "SOURCE_URL": inst.get("url", ""),
                    "SOURCE": inst.get("source", ""),
                    "FRAME_START": frame_start,
                    "FRAME_END": frame_end,
                }

                bbox = inst.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    row["BBOX_X1"] = float(bbox[0])
                    row["BBOX_Y1"] = float(bbox[1])
                    row["BBOX_X2"] = float(bbox[2])
                    row["BBOX_Y2"] = float(bbox[3])
                    row["PERSON_DETECTED"] = True
                else:
                    row["BBOX_X1"] = None
                    row["BBOX_Y1"] = None
                    row["BBOX_X2"] = None
                    row["BBOX_Y2"] = None
                    row["PERSON_DETECTED"] = False

                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=[
                "SAMPLE_ID",
                "VIDEO_ID",
                "START",
                "END",
                "GLOSS",
                "CLASS_ID",
                "SPLIT",
                "SIGNER_ID",
                "FPS",
                "VARIATION_ID",
                "SOURCE_URL",
                "SOURCE",
                "FRAME_START",
                "FRAME_END",
                "BBOX_X1",
                "BBOX_Y1",
                "BBOX_X2",
                "BBOX_Y2",
                "PERSON_DETECTED",
            ])
        else:
            df = apply_availability_policy(
                df, video_dir, source.availability_policy,
            )

        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, sep="\t", index=False)

        context.manifest_path = Path(manifest_path)
        context.manifest_df = df
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()) if not df.empty else 0,
            "segments": len(df),
        }
        self.logger.info(
            "WLASL manifest built: %d samples, %d clips -> %s",
            len(df),
            int(df["VIDEO_ID"].nunique()) if not df.empty else 0,
            manifest_path,
        )
        return context

    def _resolve_duration(
        self,
        video_dir: str,
        video_id: str,
        fps: Optional[float],
        frame_start: Optional[int],
        frame_end: Optional[int],
        available_ids,
    ) -> Optional[float]:
        if (
            fps is not None
            and fps > 0
            and frame_start is not None
            and frame_end is not None
            and frame_end > 0
            and frame_start > 0
        ):
            return round((frame_end - frame_start + 1) / fps, 6)

        if video_id not in available_ids:
            return None

        video_path = find_video_file(video_dir, video_id)
        duration = self._probe_duration_seconds(video_path)
        if duration is None or duration <= 0:
            raise RuntimeError(
                f"Unable to determine duration for WLASL clip: {video_path}"
            )
        return duration

    @staticmethod
    def _probe_duration_seconds(video_path: Path) -> Optional[float]:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                return None
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps <= 0 or frames <= 0:
                return None
            return round(float(frames) / float(fps), 6)
        except Exception:
            return None

    @staticmethod
    def _as_float(value) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int(value) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
