"""Tests for FPSSampler (video.py)."""

from signdata.utils.video import FPSSampler


class TestFPSSamplerFPSMode:
    def test_30_to_24_fps(self):
        """30→24 fps yields ~80% frames sampled."""
        sampler = FPSSampler(src_fps=30.0, sample_rate=24.0)
        assert sampler.mode == "fps"

        total = 300
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.75 <= ratio <= 0.85, f"Expected ~80%, got {ratio:.2%}"

    def test_target_gte_source_keeps_all(self):
        """target >= source keeps every frame."""
        sampler = FPSSampler(src_fps=24.0, sample_rate=30.0)
        assert sampler.mode == "fps"

        total = 100
        taken = sum(1 for _ in range(total) if sampler.take())
        assert taken == total

    def test_exact_halving(self):
        """30→15 fps yields ~50% frames sampled."""
        sampler = FPSSampler(src_fps=30.0, sample_rate=15.0)
        total = 300
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.45 <= ratio <= 0.55, f"Expected ~50%, got {ratio:.2%}"


class TestFPSSamplerRatioMode:
    def test_ratio_0_5_yields_every_other_on_average(self):
        """ratio=0.5 yields ~50% of frames."""
        sampler = FPSSampler(src_fps=30.0, sample_rate=0.5)
        assert sampler.mode == "ratio"

        total = 300
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.45 <= ratio <= 0.55, f"Expected ~50%, got {ratio:.2%}"

    def test_ratio_0_75_yields_three_quarters(self):
        sampler = FPSSampler(src_fps=40.0, sample_rate=0.75)
        assert sampler.mode == "ratio"

        total = 400
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.70 <= ratio <= 0.80, f"Expected ~75%, got {ratio:.2%}"


class TestFPSSamplerModeSelection:
    def test_none_selects_native(self):
        sampler = FPSSampler(src_fps=30.0, sample_rate=None)
        assert sampler.mode == "native"

        total = 100
        taken = sum(1 for _ in range(total) if sampler.take())
        assert taken == total

    def test_ratio_selects_ratio(self):
        sampler = FPSSampler(src_fps=30.0, sample_rate=0.75)
        assert sampler.mode == "ratio"

    def test_absolute_selects_fps(self):
        sampler = FPSSampler(src_fps=30.0, sample_rate=24.0)
        assert sampler.mode == "fps"
