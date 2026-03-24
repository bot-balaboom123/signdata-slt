"""Tests for MMPoseExtractor.__init__ parameter handling.

Only tests __init__ parameter handling — no GPU, mmpose, or real frames needed.
"""

from signdata.processors.pose.mmpose import MMPoseExtractor


class TestMMPoseInit:
    def test_default_config(self):
        """Duck-typed config creates extractor with expected attributes."""

        class _Cfg:
            bbox_threshold = 0.5
            keypoint_threshold = 0.3
            add_visible = True
            batch_size = 16

        ext = MMPoseExtractor(_Cfg())
        assert ext.bbox_threshold == 0.5
        assert ext.add_visible is True
        assert ext.det_cat_id == 0

    def test_custom_bbox_threshold(self):
        """Custom bbox_threshold is stored."""

        class _Cfg:
            bbox_threshold = 0.7
            keypoint_threshold = 0.3
            add_visible = True
            batch_size = 16

        ext = MMPoseExtractor(_Cfg())
        assert ext.bbox_threshold == 0.7

    def test_no_reduction_attributes(self):
        """Extractor no longer has reduction-related attributes."""

        class _Cfg:
            bbox_threshold = 0.5
            keypoint_threshold = 0.3
            add_visible = True
            batch_size = 16

        ext = MMPoseExtractor(_Cfg())
        assert not hasattr(ext, "apply_reduction")
        assert not hasattr(ext, "keypoint_indices")
