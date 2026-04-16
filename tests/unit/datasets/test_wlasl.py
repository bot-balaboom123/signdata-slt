"""Tests for the WLASL dataset adapter."""

import json

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.wlasl import WLASLDataset, WLASLSourceConfig
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


def _write_annotations(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_config(annotation_path, video_dir, manifest_path, source=None):
    dataset_source = {"annotation_json": str(annotation_path)}
    if source:
        dataset_source.update(source)
    return Config(
        dataset={
            "name": "wlasl",
            "source": dataset_source,
        },
        paths={
            "videos": str(video_dir),
            "manifest": str(manifest_path),
        },
    )


class TestWLASLRegistration:
    def test_registered(self):
        assert "wlasl" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert WLASLDataset().name == "wlasl"


class TestWLASLValidateConfig:
    def test_valid_config_passes(self, tmp_path):
        annotation_path = tmp_path / "WLASL_v0.3.json"
        _write_annotations(annotation_path, [])
        cfg = Config(
            dataset={
                "name": "wlasl",
                "source": {"annotation_json": str(annotation_path)},
            },
        )
        WLASLDataset.validate_config(cfg)

    def test_missing_annotation_json_raises(self):
        cfg = Config(dataset={"name": "wlasl"})
        with pytest.raises(ValueError, match="annotation_json"):
            WLASLDataset.validate_config(cfg)


class TestWLASLSourceConfig:
    def test_defaults(self, tmp_path):
        annotation_path = tmp_path / "WLASL_v0.3.json"
        _write_annotations(annotation_path, [])
        cfg = Config(
            dataset={
                "name": "wlasl",
                "source": {"annotation_json": str(annotation_path)},
            },
        )

        source = WLASLDataset().get_source_config(cfg)
        assert isinstance(source, WLASLSourceConfig)
        assert source.annotation_json == str(annotation_path)
        assert source.split == "all"
        assert source.availability_policy == "drop_unavailable"

    def test_custom_options(self, tmp_path):
        annotation_path = tmp_path / "WLASL_v0.3.json"
        _write_annotations(annotation_path, [])
        cfg = Config(
            dataset={
                "name": "wlasl",
                "source": {
                    "annotation_json": str(annotation_path),
                    "split": "train",
                    "availability_policy": "mark_unavailable",
                },
            },
        )

        source = WLASLDataset().get_source_config(cfg)
        assert source.split == "train"
        assert source.availability_policy == "mark_unavailable"


class TestWLASLDownload:
    def test_download_validates_inputs(self, tmp_path):
        annotation_path = tmp_path / "WLASL_v0.3.json"
        _write_annotations(annotation_path, [])
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(annotation_path, video_dir, manifest_path)
        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)
        assert context.stats["dataset.download"]["validated"] is True
        assert context.stats["dataset.download"]["clips_found"] == 1

    def test_download_missing_annotation_raises(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(tmp_path / "missing.json", video_dir, manifest_path)

        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        with pytest.raises(FileNotFoundError, match="annotation JSON"):
            adapter.download(cfg, context)

    def test_download_missing_video_dir_raises(self, tmp_path):
        annotation_path = tmp_path / "WLASL_v0.3.json"
        _write_annotations(annotation_path, [])
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            annotation_path,
            tmp_path / "missing-videos",
            manifest_path,
        )

        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        with pytest.raises(FileNotFoundError, match="video directory"):
            adapter.download(cfg, context)


class TestWLASLBuildManifest:
    def _make_context(self, config):
        adapter = WLASLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_build_manifest_from_annotations(self, tmp_path):
        annotation_path = tmp_path / "WLASL_v0.3.json"
        _write_annotations(annotation_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 1,
                        "frame_end": 50,
                        "signer_id": 3,
                        "variation_id": 0,
                        "source": "aslbrick",
                        "url": "https://example.com/book.mp4",
                        "bbox": [10, 20, 30, 40],
                    },
                    {
                        "video_id": "00002",
                        "split": "val",
                        "fps": 25,
                        "frame_start": 1,
                        "frame_end": 25,
                        "signer_id": 4,
                        "variation_id": 1,
                        "source": "aslbrick",
                        "url": "https://example.com/book-val.mp4",
                    },
                ],
            },
            {
                "gloss": "drink",
                "instances": [
                    {
                        "video_id": "00003",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 10,
                        "frame_end": 34,
                        "signer_id": 9,
                        "variation_id": 0,
                        "source": "signschool",
                        "url": "https://example.com/drink.mp4",
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        (video_dir / "00003.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            annotation_path,
            video_dir,
            manifest_path,
            source={"split": "train"},
        )
        adapter = WLASLDataset()
        context = self._make_context(cfg)

        context = adapter.build_manifest(cfg, context)

        assert context.manifest_path == manifest_path
        assert len(context.manifest_df) == 2
        assert list(context.manifest_df["VIDEO_ID"]) == ["00001", "00003"]
        assert list(context.manifest_df["GLOSS"]) == ["book", "drink"]
        assert list(context.manifest_df["CLASS_ID"]) == [0, 1]
        assert list(context.manifest_df["SPLIT"]) == ["train", "train"]
        assert list(context.manifest_df["END"]) == [2.0, 1.0]
        assert context.manifest_df.iloc[0]["BBOX_X1"] == 10.0
        assert context.manifest_df.iloc[0]["PERSON_DETECTED"] == True
        assert context.manifest_df.iloc[1]["PERSON_DETECTED"] == False
        assert context.stats["dataset.manifest"] == {"videos": 2, "segments": 2}

        reloaded = pd.read_csv(manifest_path, delimiter="\t")
        assert len(reloaded) == 2
        assert "GLOSS" in reloaded.columns

    def test_build_manifest_marks_unavailable_rows(self, tmp_path):
        annotation_path = tmp_path / "WLASL_v0.3.json"
        _write_annotations(annotation_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 1,
                        "frame_end": 50,
                    },
                    {
                        "video_id": "00002",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 1,
                        "frame_end": 25,
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            annotation_path,
            video_dir,
            manifest_path,
            source={"availability_policy": "mark_unavailable"},
        )

        context = self._make_context(cfg)
        context = WLASLDataset().build_manifest(cfg, context)

        assert len(context.manifest_df) == 2
        assert list(context.manifest_df["AVAILABLE"]) == [True, False]

    def test_build_manifest_probes_full_clip_duration(self, tmp_path, monkeypatch):
        annotation_path = tmp_path / "WLASL_v0.3.json"
        _write_annotations(annotation_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 1,
                        "frame_end": -1,
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(annotation_path, video_dir, manifest_path)

        monkeypatch.setattr(
            WLASLDataset,
            "_probe_duration_seconds",
            staticmethod(lambda path: 3.6),
        )

        context = self._make_context(cfg)
        context = WLASLDataset().build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["START"] == 0.0
        assert context.manifest_df.iloc[0]["END"] == 3.6
