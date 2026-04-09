"""Tests for detection utilities and config schema defaults.

Covers:
  - Pure logic helpers (union_bbox_tuples, bbox padding/clamp)
  - Schema defaults for detection/video config classes
  - Pipeline processor registration (video2pose / video2crop)
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "src").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from signdata.config.schema import (
    Config,
    PathsConfig,
    ProcessingConfig,
    VideoProcessingConfig,
    YOLODetectionConfig,
    MMDetDetectionConfig,
)
from signdata.processors.detection.validation import union_bbox_tuples
from signdata.processors.detection.yolo.backend import YOLODetector
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


# ===========================================================================
# 1. Schema defaults (config classes)
# ===========================================================================

class TestSchemaDefaults:
    def test_yolo_detection_defaults(self):
        cfg = YOLODetectionConfig()
        assert cfg.model == "yolov8n.pt"
        assert cfg.confidence_threshold == 0.5
        assert cfg.device == "cpu"
        assert cfg.min_bbox_area == 0.05

    def test_mmdet_detection_requires_paths(self):
        with pytest.raises(Exception):
            MMDetDetectionConfig()  # missing required fields

    def test_mmdet_detection_with_paths(self):
        cfg = MMDetDetectionConfig(
            det_model_config="model.py",
            det_model_checkpoint="model.pth",
        )
        assert cfg.device == "cuda:0"

    def test_video_processing_defaults(self):
        cfg = VideoProcessingConfig()
        assert cfg.codec == "libx264"
        assert cfg.padding == 0.0
        assert cfg.resize is None

    def test_paths_config_has_output_and_webdataset(self):
        p = PathsConfig()
        assert hasattr(p, "output")
        assert hasattr(p, "webdataset")
        assert p.output == ""
        assert p.webdataset == ""

    def test_config_processing_defaults(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        assert isinstance(cfg.processing, ProcessingConfig)
        assert cfg.processing.enabled is False


# ===========================================================================
# 2. union_bbox_tuples helper
# ===========================================================================

class TestUnionBboxTuples:
    def test_single_bbox(self):
        result = union_bbox_tuples([(10, 20, 100, 200)])
        assert result == (10, 20, 100, 200)

    def test_union_of_two_overlapping(self):
        b1 = (10.0, 20.0, 100.0, 200.0)
        b2 = (50.0,  5.0, 150.0, 180.0)
        x1, y1, x2, y2 = union_bbox_tuples([b1, b2])
        assert x1 == 10.0   # min x1
        assert y1 == 5.0    # min y1
        assert x2 == 150.0  # max x2
        assert y2 == 200.0  # max y2

    def test_union_of_non_overlapping(self):
        b1 = (0.0,   0.0,  50.0,  50.0)
        b2 = (60.0, 60.0, 120.0, 120.0)
        x1, y1, x2, y2 = union_bbox_tuples([b1, b2])
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == 120.0
        assert y2 == 120.0

    def test_union_identical_bboxes(self):
        b = (5.0, 10.0, 80.0, 90.0)
        assert union_bbox_tuples([b, b, b]) == b

    def test_union_five_frames(self):
        bboxes = [(i * 5.0, i * 3.0, i * 5.0 + 100.0, i * 3.0 + 200.0) for i in range(5)]
        x1, y1, x2, y2 = union_bbox_tuples(bboxes)
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == pytest.approx(4 * 5.0 + 100.0)
        assert y2 == pytest.approx(4 * 3.0 + 200.0)


# ===========================================================================
# 3. Pipeline registration
# ===========================================================================

class TestPipelineRegistration:
    def test_video2pose_registered(self):
        assert "video2pose" in PROCESSOR_REGISTRY

    def test_video2crop_registered(self):
        assert "video2crop" in PROCESSOR_REGISTRY

    def test_video2compression_registered(self):
        assert "video2compression" in PROCESSOR_REGISTRY


# ===========================================================================
# 4. YOLO init errors
# ===========================================================================

class TestYOLODetectorInit:
    def test_missing_weights_raises_clear_error(self):
        cfg = YOLODetectionConfig(model="missing-model.pt", device="cpu")

        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
            side_effect=FileNotFoundError("missing"),
        ):
            with pytest.raises(FileNotFoundError, match="YOLO weights not found"):
                YOLODetector(cfg)

    def test_cuda_unavailable_raises_clear_error(self):
        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cuda:0")

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is unavailable"):
                YOLODetector(cfg)
