"""Tests for WindowVideoProcessor.

Covers:
  - WindowVideoConfig validation and defaults
  - generate_windows() helper — various time ranges, strides, edge cases
  - Processor run() with timing-based manifests
  - Processor run() with no-timing manifests (mocked video duration)
  - Stage manifest output format and context updates
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "src").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signdata.config.schema import Config
from signdata.processors.video.window import (
    WindowVideoConfig,
    WindowVideoProcessor,
    generate_windows,
    _get_video_duration,
)
from signdata.pipeline.context import PipelineContext
from signdata.datasets.youtube_asl import YouTubeASLDataset
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


# ===========================================================================
# 1. WindowVideoConfig
# ===========================================================================

class TestWindowVideoConfig:
    def test_defaults(self):
        cfg = WindowVideoConfig()
        assert cfg.window_seconds == 10.0
        assert cfg.stride_seconds == 5.0
        assert cfg.min_window_seconds == 2.0
        assert cfg.align_to_captions is False

    def test_custom_values(self):
        cfg = WindowVideoConfig(
            window_seconds=30.0,
            stride_seconds=15.0,
            min_window_seconds=5.0,
        )
        assert cfg.window_seconds == 30.0
        assert cfg.stride_seconds == 15.0
        assert cfg.min_window_seconds == 5.0

    def test_zero_stride_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            WindowVideoConfig(stride_seconds=0.0)

    def test_negative_stride_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            WindowVideoConfig(stride_seconds=-1.0)


# ===========================================================================
# 2. generate_windows()
# ===========================================================================

class TestGenerateWindows:
    def test_basic_windows(self):
        """18s video, 10s window, 5s stride → 4 windows."""
        windows = generate_windows(
            video_id="vid1",
            start=0.0, end=18.0,
            window_sec=10.0, stride_sec=5.0, min_sec=2.0,
            shared_meta={"SPLIT": "train"},
        )
        assert len(windows) == 4
        assert windows[0]["SAMPLE_ID"] == "vid1_w000"
        assert windows[0]["START"] == 0.0
        assert windows[0]["END"] == 10.0
        assert windows[1]["START"] == 5.0
        assert windows[1]["END"] == 15.0
        assert windows[2]["START"] == 10.0
        assert windows[2]["END"] == 18.0  # clamped to video end
        assert windows[3]["START"] == 15.0
        assert windows[3]["END"] == 18.0  # 3s >= 2s min

    def test_ingestion_metadata_copied(self):
        windows = generate_windows(
            video_id="vid1",
            start=0.0, end=10.0,
            window_sec=10.0, stride_sec=10.0, min_sec=1.0,
            shared_meta={"SPLIT": "train", "SIGNER_ID": "signer_42"},
        )
        assert len(windows) == 1
        assert windows[0]["SPLIT"] == "train"
        assert windows[0]["SIGNER_ID"] == "signer_42"

    def test_no_windows_for_short_video(self):
        """Video shorter than min_window_seconds → no windows."""
        windows = generate_windows(
            video_id="vid1",
            start=0.0, end=1.0,
            window_sec=10.0, stride_sec=5.0, min_sec=2.0,
            shared_meta={},
        )
        assert windows == []

    def test_trailing_short_window_dropped(self):
        """Trailing window shorter than min_sec should be dropped."""
        # 11s video, 10s window, 10s stride → window at [0,10), trailing [10,11)=1s < 2s min
        windows = generate_windows(
            video_id="vid1",
            start=0.0, end=11.0,
            window_sec=10.0, stride_sec=10.0, min_sec=2.0,
            shared_meta={},
        )
        assert len(windows) == 1
        assert windows[0]["END"] == 10.0

    def test_non_zero_start(self):
        """Windows within a sub-range of a video."""
        windows = generate_windows(
            video_id="vid1",
            start=5.0, end=25.0,
            window_sec=10.0, stride_sec=10.0, min_sec=1.0,
            shared_meta={},
        )
        assert len(windows) == 2
        assert windows[0]["START"] == 5.0
        assert windows[0]["END"] == 15.0
        assert windows[1]["START"] == 15.0
        assert windows[1]["END"] == 25.0

    def test_stride_larger_than_window(self):
        """Stride > window → gaps between windows (no overlap)."""
        windows = generate_windows(
            video_id="vid1",
            start=0.0, end=30.0,
            window_sec=5.0, stride_sec=10.0, min_sec=1.0,
            shared_meta={},
        )
        assert len(windows) == 3
        assert windows[0]["START"] == 0.0
        assert windows[0]["END"] == 5.0
        assert windows[1]["START"] == 10.0
        assert windows[1]["END"] == 15.0
        assert windows[2]["START"] == 20.0
        assert windows[2]["END"] == 25.0

    def test_video_id_in_sample_id(self):
        """SAMPLE_ID should use the VIDEO_ID as prefix."""
        windows = generate_windows(
            video_id="my_video",
            start=0.0, end=10.0,
            window_sec=5.0, stride_sec=5.0, min_sec=1.0,
            shared_meta={},
        )
        assert windows[0]["SAMPLE_ID"] == "my_video_w000"
        assert windows[1]["SAMPLE_ID"] == "my_video_w001"

    def test_exact_division(self):
        """20s video, 10s window, 10s stride → exactly 2 windows."""
        windows = generate_windows(
            video_id="vid1",
            start=0.0, end=20.0,
            window_sec=10.0, stride_sec=10.0, min_sec=1.0,
            shared_meta={},
        )
        assert len(windows) == 2


# ===========================================================================
# 3. Processor run() — manifest with timing
# ===========================================================================

class TestWindowVideoProcessorWithTiming:
    def _make_context(self, cfg, manifest_path, tmp_path):
        return PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            manifest_path=manifest_path,
            videos_dir=tmp_path / "videos",
            output_dir=tmp_path / "window_video" / "default",
        )

    def test_run_with_timing_manifest(self, tmp_path):
        """Processor creates windowed manifest from timed input."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a", "vid_a", "vid_b"],
            "SAMPLE_ID": ["vid_a-0", "vid_a-1", "vid_b-0"],
            "START": [0.0, 10.0, 0.0],
            "END": [10.0, 25.0, 18.0],
            "TEXT": ["hello", "world", "test"],
            "SPLIT": ["train", "train", "val"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={"root": str(tmp_path)},
        )

        processor = WindowVideoProcessor(cfg)
        ctx = self._make_context(cfg, manifest_path, tmp_path)

        # Video files unavailable → falls back to max(END) per VIDEO_ID
        with patch(
            "signdata.processors.video.window._get_video_duration",
            return_value=0.0,
        ):
            ctx = processor.run(ctx)

        # Stage manifest should be written
        stage_manifest = tmp_path / "window_video" / "default" / "manifest.csv"
        assert stage_manifest.exists()

        result = pd.read_csv(stage_manifest, sep="\t")
        # vid_a spans 0-25, vid_b spans 0-18
        # All windows should have VIDEO_ID, SAMPLE_ID, START, END
        assert "VIDEO_ID" in result.columns
        assert "SAMPLE_ID" in result.columns
        assert "START" in result.columns
        assert "END" in result.columns
        # Labels should NOT be carried over
        assert "TEXT" not in result.columns
        # Shared metadata should be carried over
        assert "SPLIT" in result.columns

        # context.manifest_df should be updated
        assert ctx.manifest_df is not None
        assert len(ctx.manifest_df) == len(result)

        # Stats
        assert ctx.stats["window_video"]["total"] > 0
        assert ctx.stats["window_video"]["source_rows"] == 3

    def test_timed_manifest_uses_full_video_duration(self, tmp_path):
        """When videos are readable, windows span the full video duration."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["vid_a-0"],
            "START": [5.0],       # Caption starts at 5s (not 0)
            "END": [20.0],        # Caption ends at 20s
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        (tmp_path / "videos").mkdir()

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={"root": str(tmp_path), "videos": str(tmp_path / "videos")},
        )
        processor = WindowVideoProcessor(cfg)
        ctx = self._make_context(cfg, manifest_path, tmp_path)
        ctx.videos_dir = tmp_path / "videos"

        # Mock duration to 30s (longer than caption span 5-20)
        with patch(
            "signdata.processors.video.window._get_video_duration",
            return_value=30.0,
        ):
            ctx = processor.run(ctx)

        result = ctx.manifest_df
        # Default stride=5s, window=10s over [0,30):
        # [0,10), [5,15), [10,20), [15,25), [20,30), [25,30) = 6 windows
        assert len(result) == 6
        assert result.iloc[0]["START"] == 0.0
        assert result.iloc[0]["END"] == 10.0
        assert result.iloc[-1]["START"] == 25.0
        assert result.iloc[-1]["END"] == 30.0

    def test_timed_manifest_falls_back_to_max_end(self, tmp_path):
        """When video is unreadable, timed manifest falls back to max(END)."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["vid_a-0"],
            "START": [5.0],
            "END": [20.0],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={"root": str(tmp_path)},
        )
        processor = WindowVideoProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            manifest_path=manifest_path,
            videos_dir=None,
            output_dir=tmp_path / "window_video" / "default",
        )

        ctx = processor.run(ctx)

        result = ctx.manifest_df
        # Falls back to max(END)=20.0, start=0.0, default stride=5s:
        # [0,10), [5,15), [10,20), [15,20) = 4 windows
        assert len(result) == 4
        assert result.iloc[0]["START"] == 0.0
        assert result.iloc[0]["END"] == 10.0
        assert result.iloc[-1]["START"] == 15.0
        assert result.iloc[-1]["END"] == 20.0

    def test_label_columns_dropped(self, tmp_path):
        """TEXT, GLOSS, CLASS_ID should not appear in windowed manifest."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["vid_a-0"],
            "START": [0.0],
            "END": [20.0],
            "TEXT": ["hello world"],
            "GLOSS": ["HELLO WORLD"],
            "CLASS_ID": ["greeting"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={"root": str(tmp_path)},
        )
        processor = WindowVideoProcessor(cfg)
        ctx = self._make_context(cfg, manifest_path, tmp_path)

        with patch(
            "signdata.processors.video.window._get_video_duration",
            return_value=0.0,
        ):
            ctx = processor.run(ctx)

        result = ctx.manifest_df
        assert "TEXT" not in result.columns
        assert "GLOSS" not in result.columns
        assert "CLASS_ID" not in result.columns


# ===========================================================================
# 4. Processor run() — manifest without timing (mocked video duration)
# ===========================================================================

class TestWindowVideoProcessorNoTiming:
    def test_run_reads_video_duration(self, tmp_path):
        """Without START/END, processor gets duration from video files."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a", "vid_b"],
            "SAMPLE_ID": ["vid_a", "vid_b"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        (tmp_path / "videos").mkdir()

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={
                "root": str(tmp_path),
                "videos": str(tmp_path / "videos"),
            },
        )
        processor = WindowVideoProcessor(cfg)

        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            manifest_path=manifest_path,
            videos_dir=tmp_path / "videos",
            output_dir=tmp_path / "window_video" / "default",
        )

        # Mock _get_video_duration to return known durations
        with patch(
            "signdata.processors.video.window._get_video_duration",
            side_effect=lambda path: 25.0 if "vid_a" in path else 15.0,
        ):
            ctx = processor.run(ctx)

        result = ctx.manifest_df
        assert result is not None

        # vid_a (25s), default stride=5s: [0,10),[5,15),[10,20),[15,25),[20,25) → 5 windows
        vid_a_windows = result[result["VIDEO_ID"] == "vid_a"]
        assert len(vid_a_windows) == 5

        # vid_b (15s), default stride=5s: [0,10),[5,15),[10,15) → 3 windows
        vid_b_windows = result[result["VIDEO_ID"] == "vid_b"]
        assert len(vid_b_windows) == 3

    def test_raises_when_all_videos_unreadable(self, tmp_path):
        """When no windows can be generated, processor raises RuntimeError."""
        manifest_path = tmp_path / "manifest.csv"
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["vid_a"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        (tmp_path / "videos").mkdir()

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={
                "root": str(tmp_path),
                "videos": str(tmp_path / "videos"),
            },
        )
        processor = WindowVideoProcessor(cfg)

        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            manifest_path=manifest_path,
            videos_dir=tmp_path / "videos",
            output_dir=tmp_path / "window_video" / "default",
        )

        with patch(
            "signdata.processors.video.window._get_video_duration",
            return_value=0.0,
        ):
            with pytest.raises(RuntimeError, match="produced no windows"):
                processor.run(ctx)


# ===========================================================================
# 5. validate_inputs
# ===========================================================================

class TestWindowVideoValidateInputs:
    def test_raises_when_manifest_missing(self, tmp_path):
        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={"root": str(tmp_path)},
        )
        processor = WindowVideoProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            manifest_path=tmp_path / "nonexistent.csv",
        )
        with pytest.raises(RuntimeError, match="manifest not found"):
            processor.validate_inputs(ctx)

    def test_raises_when_video_dir_missing_untimed(self, tmp_path):
        """Untimed manifests require video_dir to read durations."""
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["vid_a"],
        }).to_csv(manifest_path, sep="\t", index=False)

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={"root": str(tmp_path)},
        )
        processor = WindowVideoProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            manifest_path=manifest_path,
            videos_dir=tmp_path / "nonexistent_videos",
        )
        with pytest.raises(RuntimeError, match="video directory"):
            processor.validate_inputs(ctx)

    def test_no_raise_when_video_dir_missing_timed(self, tmp_path):
        """Timed manifests don't require video_dir (can fall back to max END)."""
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["vid_a"],
            "START": [0.0],
            "END": [20.0],
        }).to_csv(manifest_path, sep="\t", index=False)

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={"root": str(tmp_path)},
        )
        processor = WindowVideoProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
            manifest_path=manifest_path,
            videos_dir=tmp_path / "nonexistent_videos",
        )
        # Should NOT raise — timed manifests can fall back to max(END)
        processor.validate_inputs(ctx)


# ===========================================================================
# 6. Registration
# ===========================================================================

class TestWindowVideoRegistration:
    def test_registered(self):
        assert "window_video" in PROCESSOR_REGISTRY
