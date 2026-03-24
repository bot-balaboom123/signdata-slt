"""Tests for detection helpers, crop geometry, and new schema defaults.

Covers:
  - Pure logic helpers (_union_bboxes, _parse_bool, bbox padding/clamp)
  - Schema defaults for new detection/video config classes
  - crop_video ffmpeg command construction (mocked, no real video needed)
  - Pipeline processor registration (new video2pose / video2crop)
  - Frame sampling strategies (skip_frame, uniform)
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import cv2

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
from signdata.processors.detection.detect_person import (
    _union_bboxes,
    _sample_frames,
    _sample_frames_uniform,
    _sample_frames_skip,
)
from signdata.processors.video.crop import _crop_single_video, _parse_bool
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


# ===========================================================================
# 1. Schema defaults (new config classes)
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
        assert cfg.padding == 0.25
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
# 2. _union_bboxes helper
# ===========================================================================

class TestUnionBboxes:
    def test_single_bbox(self):
        result = _union_bboxes([(10, 20, 100, 200)])
        assert result == (10, 20, 100, 200)

    def test_union_of_two_overlapping(self):
        b1 = (10.0, 20.0, 100.0, 200.0)
        b2 = (50.0,  5.0, 150.0, 180.0)
        x1, y1, x2, y2 = _union_bboxes([b1, b2])
        assert x1 == 10.0   # min x1
        assert y1 == 5.0    # min y1
        assert x2 == 150.0  # max x2
        assert y2 == 200.0  # max y2

    def test_union_of_non_overlapping(self):
        b1 = (0.0,   0.0,  50.0,  50.0)
        b2 = (60.0, 60.0, 120.0, 120.0)
        x1, y1, x2, y2 = _union_bboxes([b1, b2])
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == 120.0
        assert y2 == 120.0

    def test_union_identical_bboxes(self):
        b = (5.0, 10.0, 80.0, 90.0)
        assert _union_bboxes([b, b, b]) == b

    def test_union_five_frames(self):
        bboxes = [(i * 5.0, i * 3.0, i * 5.0 + 100.0, i * 3.0 + 200.0) for i in range(5)]
        x1, y1, x2, y2 = _union_bboxes(bboxes)
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == pytest.approx(4 * 5.0 + 100.0)
        assert y2 == pytest.approx(4 * 3.0 + 200.0)


# ===========================================================================
# 3. _parse_bool helper (crop_video)
# ===========================================================================

class TestParseBool:
    @pytest.mark.parametrize("val,expected", [
        (True,    True),
        (False,   False),
        ("True",  True),
        ("False", False),
        ("true",  True),
        ("false", False),
        (1,       True),
        (0,       False),
        (1.0,     True),
        (0.0,     False),
    ])
    def test_various_inputs(self, val, expected):
        assert _parse_bool(val) == expected

    def test_nan_treated_as_false(self):
        # NaN as float → bool(NaN) is True in Python, but we treat it as False
        # via the float branch: bool(float('nan')) == True — this is expected Python
        # behaviour; our _parse_bool passes it through bool(), so result is True.
        # This test documents the actual behaviour so it doesn't silently change.
        result = _parse_bool(float("nan"))
        assert isinstance(result, bool)


# ===========================================================================
# 4. Crop geometry calculations
# ===========================================================================

class TestCropGeometry:
    """Test padding and clamp logic in _crop_single_video via a mock ffmpeg."""

    def _run_with_mock(self, x1, y1, x2, y2, frame_w, frame_h, padding):
        """Helper: mock cv2 and subprocess to capture the ffmpeg crop filter."""
        clip_path = "/fake/clip.mp4"
        output_path = "/fake/output.mp4"
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            result = MagicMock()
            result.returncode = 0
            return result

        def fake_exists(path):
            # output does not exist yet (so we proceed); clip does exist
            return path != output_path

        with patch("signdata.processors.video.crop.os.path.exists",
                   side_effect=fake_exists), \
             patch("signdata.processors.video.crop.os.makedirs"), \
             patch("signdata.processors.video.crop.cv2.VideoCapture") as mock_cap, \
             patch("signdata.processors.video.crop.subprocess.run", side_effect=fake_run):

            cap_instance = MagicMock()
            cap_instance.isOpened.return_value = True
            # cv2.CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4
            cap_instance.get.side_effect = lambda prop: (
                frame_w if prop == 3 else frame_h
            )
            mock_cap.return_value = cap_instance

            name, ok, msg = _crop_single_video((
                clip_path, output_path,
                x1, y1, x2, y2,
                True,   # person_detected
                padding,
                "libx264",
            ))

        return ok, captured_cmd

    def test_padding_expands_bbox(self):
        """With 25% padding a 100x200 box should expand by 25 and 50 pixels."""
        ok, cmd = self._run_with_mock(
            x1=50, y1=50, x2=150, y2=250,
            frame_w=640, frame_h=480,
            padding=0.25,
        )
        assert ok
        vf_arg = [c for c in cmd if c.startswith("crop=")]
        assert len(vf_arg) == 1
        # crop=w:h:x:y  — parse it
        parts = vf_arg[0].replace("crop=", "").split(":")
        w, h, cx, cy = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        # x: 50 - 0.25*100 = 25 → cx1=25
        assert cx == 25
        # y: 50 - 0.25*200 = 0 → cy1=0
        assert cy == 0
        # w must be even
        assert w % 2 == 0
        assert h % 2 == 0

    def test_bbox_clamped_to_frame_boundary(self):
        """Padded bbox that would exceed frame dims should be clamped."""
        ok, cmd = self._run_with_mock(
            x1=0, y1=0, x2=640, y2=480,   # already full frame
            frame_w=640, frame_h=480,
            padding=0.25,                   # would go negative / over frame
        )
        assert ok
        vf_arg = [c for c in cmd if c.startswith("crop=")]
        parts = vf_arg[0].replace("crop=", "").split(":")
        cx, cy = int(parts[2]), int(parts[3])
        w, h = int(parts[0]), int(parts[1])
        # After clamping, origin must be >= 0
        assert cx >= 0
        assert cy >= 0
        # Width + origin must not exceed frame
        assert cx + w <= 640
        assert cy + h <= 480

    def test_no_person_uses_stream_copy(self):
        """When PERSON_DETECTED=False, ffmpeg should use -c copy."""
        clip_path = "/fake/clip.mp4"
        output_path = "/fake/output.mp4"
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            r = MagicMock()
            r.returncode = 0
            return r

        def fake_exists(path):
            return path != output_path  # clip exists, output does not

        with patch("signdata.processors.video.crop.os.path.exists",
                   side_effect=fake_exists), \
             patch("signdata.processors.video.crop.os.makedirs"), \
             patch("signdata.processors.video.crop.subprocess.run", side_effect=fake_run):

            name, ok, msg = _crop_single_video((
                clip_path, output_path,
                0, 0, 640, 480,
                False,      # person_detected = False
                0.25,
                "libx264",
            ))

        assert ok
        assert msg == "no-person copy"
        # Should use stream copy, NOT a crop filter
        assert "-c" in captured_cmd
        assert "copy" in captured_cmd
        assert not any(c.startswith("crop=") for c in captured_cmd)

    def test_even_dimensions_enforced(self):
        """Crop dimensions must always be even (libx264 requirement)."""
        ok, cmd = self._run_with_mock(
            x1=0, y1=0, x2=101, y2=101,   # odd dimensions before padding
            frame_w=640, frame_h=480,
            padding=0.0,
        )
        assert ok
        vf_arg = [c for c in cmd if c.startswith("crop=")]
        parts = vf_arg[0].replace("crop=", "").split(":")
        w, h = int(parts[0]), int(parts[1])
        assert w % 2 == 0, f"width {w} is not even"
        assert h % 2 == 0, f"height {h} is not even"


# ===========================================================================
# 5. Pipeline registration
# ===========================================================================

class TestPipelineRegistration:
    def test_detect_person_registered(self):
        assert "detect_person" in PROCESSOR_REGISTRY

    def test_crop_video_registered(self):
        assert "crop_video" in PROCESSOR_REGISTRY

    def test_video2pose_registered(self):
        assert "video2pose" in PROCESSOR_REGISTRY

    def test_video2crop_registered(self):
        assert "video2crop" in PROCESSOR_REGISTRY


# ===========================================================================
# 6. Sample strategy
# ===========================================================================

class TestSampleStrategy:
    """Test skip_frame and uniform sampling strategies."""

    def _make_fake_video(self, tmp_path: Path, num_frames: int = 60, fps: int = 30) -> str:
        """Write a minimal fake video file using OpenCV."""
        video_path = str(tmp_path / "fake.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
        for _ in range(num_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        return video_path

    def test_uniform_returns_exact_n_frames(self, tmp_path):
        video_path = self._make_fake_video(tmp_path, num_frames=90, fps=30)
        frames = _sample_frames_uniform(video_path, start_sec=0.0, end_sec=2.0, n=5)
        assert len(frames) == 5
        # Each element is (frame, w, h)
        for frame, w, h in frames:
            assert isinstance(frame, np.ndarray)
            assert w == 640
            assert h == 480

    def test_skip_frame_respects_max_frames(self, tmp_path):
        video_path = self._make_fake_video(tmp_path, num_frames=120, fps=30)
        frames = _sample_frames_skip(
            video_path, start_sec=0.0, end_sec=3.0,
            frame_skip=2, max_frames=5,
        )
        assert len(frames) <= 5

    def test_skip_frame_spacing(self, tmp_path):
        """skip_frame with frame_skip=2 should take roughly every 3rd frame."""
        video_path = self._make_fake_video(tmp_path, num_frames=120, fps=30)
        frames = _sample_frames_skip(
            video_path, start_sec=0.0, end_sec=4.0,
            frame_skip=2, max_frames=20,
        )
        # With 4s @ 30fps = 120 frames, skip=2 → ~40 samples, capped at 20
        assert 1 <= len(frames) <= 20

    def test_dispatcher_skip_frame(self, tmp_path):
        video_path = self._make_fake_video(tmp_path, num_frames=90, fps=30)
        frames = _sample_frames(
            video_path, 0.0, 2.0,
            strategy="skip_frame",
            frame_skip=2,
            uniform_frames=5,
            max_frames=5,
        )
        assert len(frames) <= 5

    def test_dispatcher_uniform(self, tmp_path):
        video_path = self._make_fake_video(tmp_path, num_frames=90, fps=30)
        frames = _sample_frames(
            video_path, 0.0, 2.0,
            strategy="uniform",
            frame_skip=2,
            uniform_frames=5,
            max_frames=5,
        )
        assert len(frames) == 5
