"""Pydantic configuration models."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# --- Detection configs (one per backend, no `type` field) ---

class YOLODetectionConfig(BaseModel):
    model: str = "yolov8n.pt"
    device: str = "cpu"
    confidence_threshold: float = 0.5
    min_bbox_area: float = 0.05


class MMDetDetectionConfig(BaseModel):
    det_model_config: str
    det_model_checkpoint: str
    device: str = "cuda:0"


class MediaPipeDetectionConfig(BaseModel):
    min_detection_confidence: float = 0.5


# --- Pose configs (one per backend, no `type` field) ---

class MediaPipePoseConfig(BaseModel):
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_face_landmarks: bool = True
    batch_size: int = 16


class MMPosePoseConfig(BaseModel):
    pose_model_config: str
    pose_model_checkpoint: str
    device: str = "cuda:0"
    bbox_threshold: float = 0.5
    keypoint_threshold: float = 0.3
    batch_size: int = 16


# --- Config selector maps (used by model_validator) ---

DETECTION_CONFIG_MAP = {
    "yolo": YOLODetectionConfig,
    "mmdet": MMDetDetectionConfig,
    "mediapipe": MediaPipeDetectionConfig,
    "null": None,
}

POSE_CONFIG_MAP = {
    "mediapipe": MediaPipePoseConfig,
    "mmpose": MMPosePoseConfig,
}


# --- Video processing config ---

class VideoProcessingConfig(BaseModel):
    codec: str = "libx264"
    padding: float = 0.25
    resize: Optional[List[int]] = None


# --- Stage configs ---

class DatasetConfig(BaseModel):
    name: str
    download: bool = True
    manifest: bool = True
    source: Dict[str, Any] = {}


class ProcessingConfig(BaseModel):
    enabled: bool = True
    processor: Literal["video2pose", "video2crop"] = "video2pose"
    detection: Literal["yolo", "mediapipe", "mmdet", "null"] = "null"
    pose: Optional[Literal["mediapipe", "mmpose"]] = None

    # Shared params
    frame_skip: int = 2
    target_fps: Optional[float] = 24.0
    max_workers: int = 1

    # Backend-specific configs — raw dicts from YAML, validated into typed
    # models by model_validator and stored back as public attributes.
    detection_config: Optional[Union[
        YOLODetectionConfig, MMDetDetectionConfig,
        MediaPipeDetectionConfig, dict,
    ]] = None
    pose_config: Optional[Union[
        MediaPipePoseConfig, MMPosePoseConfig, dict,
    ]] = None
    video_config: Optional[VideoProcessingConfig] = None

    @model_validator(mode="after")
    def validate_backend_configs(self):
        """Parse raw dicts into typed config models."""
        # Skip validation when processing is disabled (allows default construction)
        if not self.enabled:
            return self

        # Validate + replace detection_config
        det_cls = DETECTION_CONFIG_MAP.get(self.detection)
        if det_cls and self.detection_config is not None:
            if isinstance(self.detection_config, dict):
                self.detection_config = det_cls(**self.detection_config)
        elif det_cls and self.detection_config is None:
            raise ValueError(
                f"detection={self.detection!r} requires detection_config"
            )

        # video2pose requires pose + pose_config
        if self.processor == "video2pose":
            if not self.pose:
                raise ValueError("processing.pose is required for video2pose")
            pose_cls = POSE_CONFIG_MAP.get(self.pose)
            if pose_cls and self.pose_config is not None:
                if isinstance(self.pose_config, dict):
                    self.pose_config = pose_cls(**self.pose_config)
            elif pose_cls and self.pose_config is None:
                raise ValueError(
                    f"pose={self.pose!r} requires pose_config"
                )

        # video2crop requires video_config (defaults if omitted)
        if self.processor == "video2crop" and not self.video_config:
            self.video_config = VideoProcessingConfig()

        return self


class NormalizeConfig(BaseModel):
    mode: Literal["isotropic_3d", "xy_isotropic_z_minmax"] = "xy_isotropic_z_minmax"
    remove_z: bool = False
    select_keypoints: bool = True
    keypoint_preset: Optional[str] = None
    keypoint_indices: Optional[List[int]] = None
    mask_empty_frames: bool = True
    mask_low_confidence: bool = False
    visibility_threshold: float = 0.3
    missing_value: float = -999.0

    @field_validator("keypoint_preset")
    @classmethod
    def validate_keypoint_preset(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            from signdata.processors.pose import KEYPOINT_PRESETS
            if v not in KEYPOINT_PRESETS:
                raise ValueError(
                    f"Unknown keypoint preset '{v}'. "
                    f"Available: {sorted(KEYPOINT_PRESETS.keys())}"
                )
        return v


class PostProcessingConfig(BaseModel):
    enabled: bool = True
    recipes: List[str] = []
    normalize: Optional[NormalizeConfig] = None


class OutputConfig(BaseModel):
    enabled: bool = True
    type: Literal["webdataset"] = "webdataset"
    config: Dict[str, Any] = {}


class PathsConfig(BaseModel):
    root: str = ""
    videos: str = ""
    transcripts: str = ""
    manifest: str = ""
    output: str = ""
    webdataset: str = ""


# --- Top-level ---

class Config(BaseModel):
    dataset: DatasetConfig
    processing: ProcessingConfig = ProcessingConfig(enabled=False)
    post_processing: PostProcessingConfig = PostProcessingConfig()
    output: OutputConfig = OutputConfig()
    run_name: str = "default"
    paths: PathsConfig = PathsConfig()
