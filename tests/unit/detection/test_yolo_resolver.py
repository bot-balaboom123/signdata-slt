"""Tests for YOLO model resolver: alias validation, typo suggestions, cache logic."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "src").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signdata.processors.detection.yolo.resolver import (
    VALID_ALIASES,
    SUPPORTED_FAMILIES,
    is_valid_alias,
    resolve_yolo_model,
)
from signdata.config.schema import YOLODetectionConfig


# ===========================================================================
# 1. Alias catalogue
# ===========================================================================

class TestAliasCatalogue:
    def test_yolov8_aliases(self):
        for size in ["n", "s", "m", "l", "x"]:
            assert f"yolov8{size}.pt" in VALID_ALIASES

    def test_yolov9_aliases(self):
        for size in ["t", "s", "m", "c", "e"]:
            assert f"yolov9{size}.pt" in VALID_ALIASES

    def test_yolo11_aliases(self):
        for size in ["n", "s", "m", "l", "x"]:
            assert f"yolo11{size}.pt" in VALID_ALIASES

    def test_yolo26_aliases(self):
        for size in ["n", "s", "m", "l", "x"]:
            assert f"yolo26{size}.pt" in VALID_ALIASES

    def test_invalid_alias_not_in_set(self):
        assert "yolov10n.pt" not in VALID_ALIASES
        assert "yolov11m.pt" not in VALID_ALIASES  # typo form
        assert "yolov26n.pt" not in VALID_ALIASES  # typo form

    def test_is_valid_alias_with_pt(self):
        assert is_valid_alias("yolov8n.pt") is True
        assert is_valid_alias("yolo11m.pt") is True
        assert is_valid_alias("yolo26n.pt") is True
        assert is_valid_alias("yolov11m.pt") is False
        assert is_valid_alias("random.pt") is False

    def test_is_valid_alias_bare_stem(self):
        """Bare stems (without .pt) should also be recognized."""
        assert is_valid_alias("yolov8n") is True
        assert is_valid_alias("yolo11m") is True
        assert is_valid_alias("yolo26n") is True
        assert is_valid_alias("yolov26n") is False


# ===========================================================================
# 2. Local file resolution
# ===========================================================================

class TestLocalFileResolution:
    def test_existing_local_file(self, tmp_path):
        weights = tmp_path / "my_custom_model.pt"
        weights.touch()
        result = resolve_yolo_model(str(weights))
        assert result == str(weights)

    def test_cached_in_weights_dir(self, tmp_path):
        """When a valid alias is found in weights_dir, return the cached path."""
        weights_dir = tmp_path / "cache"
        weights_dir.mkdir()
        (weights_dir / "yolo11n.pt").touch()

        # Patch Path.is_file so the bare alias "yolo11n.pt" is NOT found in cwd
        original_is_file = Path.is_file
        def mock_is_file(self):
            if str(self) == "yolo11n.pt":
                return False  # not in cwd
            return original_is_file(self)

        with patch.object(Path, "is_file", mock_is_file):
            result = resolve_yolo_model("yolo11n.pt", weights_dir=str(weights_dir))
        assert result == str(weights_dir / "yolo11n.pt")


# ===========================================================================
# 3. Typo detection
# ===========================================================================

class TestTypoDetection:
    def test_yolov11_suggests_yolo11(self):
        with pytest.raises(ValueError, match="Did you mean 'yolo11m.pt'"):
            resolve_yolo_model("yolov11m.pt")

    def test_yolov11n_suggests_yolo11n(self):
        with pytest.raises(ValueError, match="Did you mean 'yolo11n.pt'"):
            resolve_yolo_model("yolov11n.pt")

    def test_yolov11x_suggests_yolo11x(self):
        with pytest.raises(ValueError, match="Did you mean 'yolo11x.pt'"):
            resolve_yolo_model("yolov11x.pt")

    def test_yolov26n_suggests_yolo26n(self):
        with pytest.raises(ValueError, match="Did you mean 'yolo26n.pt'"):
            resolve_yolo_model("yolov26n.pt")

    def test_yolov26m_suggests_yolo26m(self):
        with pytest.raises(ValueError, match="Did you mean 'yolo26m.pt'"):
            resolve_yolo_model("yolov26m.pt")

    def test_bare_stem_typo(self):
        """Bare stem typo should also be caught."""
        with pytest.raises(ValueError, match="Did you mean 'yolo26n.pt'"):
            resolve_yolo_model("yolov26n")


# ===========================================================================
# 4. Valid alias with download enabled
# ===========================================================================

class TestValidAliasDownload:
    def test_valid_alias_returns_alias(self):
        result = resolve_yolo_model("yolov8n.pt", allow_download=True)
        assert result == "yolov8n.pt"

    def test_yolo11_alias_returns_alias(self):
        result = resolve_yolo_model("yolo11m.pt", allow_download=True)
        assert result == "yolo11m.pt"

    def test_yolov9_alias_returns_alias(self):
        result = resolve_yolo_model("yolov9c.pt", allow_download=True)
        assert result == "yolov9c.pt"

    def test_yolo26_alias_returns_alias(self):
        result = resolve_yolo_model("yolo26n.pt", allow_download=True)
        assert result == "yolo26n.pt"

    def test_bare_stem_normalised_to_pt(self):
        """Bare stem input should resolve to .pt form."""
        result = resolve_yolo_model("yolov8n", allow_download=True)
        assert result == "yolov8n.pt"

    def test_yolo26_bare_stem(self):
        result = resolve_yolo_model("yolo26n", allow_download=True)
        assert result == "yolo26n.pt"


# ===========================================================================
# 5. Remote upstream-managed model references
# ===========================================================================

class TestRemoteModelReferences:
    def test_remote_weights_url_passes_through(self):
        result = resolve_yolo_model("https://example.com/models/yolo11m.pt")
        assert result == "https://example.com/models/yolo11m.pt"

    def test_remote_weights_url_requires_download_when_disabled(self):
        with pytest.raises(FileNotFoundError, match="remote weights URL"):
            resolve_yolo_model(
                "https://example.com/models/yolo11m.pt",
                allow_download=False,
            )

    def test_hub_model_passes_through(self):
        result = resolve_yolo_model("https://hub.ultralytics.com/models/ABC123")
        assert result == "https://hub.ultralytics.com/models/ABC123"

    def test_hub_model_requires_download_when_disabled(self):
        with pytest.raises(FileNotFoundError, match="Ultralytics HUB reference"):
            resolve_yolo_model(
                "https://hub.ultralytics.com/models/ABC123",
                allow_download=False,
            )

    def test_triton_model_passes_through(self):
        result = resolve_yolo_model("grpc://localhost:8000/v2/models/yolo11m")
        assert result == "grpc://localhost:8000/v2/models/yolo11m"


# ===========================================================================
# 6. allow_download=False
# ===========================================================================

class TestDownloadDisabled:
    def test_alias_no_cache_raises(self):
        """Valid alias + allow_download=False + no cached file => FileNotFoundError."""
        original_is_file = Path.is_file
        def mock_is_file(self):
            if self.name == "yolo11x.pt":
                return False
            return original_is_file(self)

        with patch.object(Path, "is_file", mock_is_file):
            with pytest.raises(FileNotFoundError, match="allow_download=False"):
                resolve_yolo_model(
                    "yolo11x.pt",
                    allow_download=False,
                    weights_dir="/nonexistent/path",
                )

    def test_alias_cached_in_weights_dir(self, tmp_path):
        weights_dir = tmp_path / "cache"
        weights_dir.mkdir()
        (weights_dir / "yolo11m.pt").touch()

        original_is_file = Path.is_file
        def mock_is_file(self):
            if str(self) == "yolo11m.pt":
                return False  # not in cwd
            return original_is_file(self)

        with patch.object(Path, "is_file", mock_is_file):
            result = resolve_yolo_model(
                "yolo11m.pt",
                allow_download=False,
                weights_dir=str(weights_dir),
            )
        assert result == str(weights_dir / "yolo11m.pt")

    def test_yolo26_cached_in_weights_dir(self, tmp_path):
        weights_dir = tmp_path / "cache"
        weights_dir.mkdir()
        (weights_dir / "yolo26n.pt").touch()

        original_is_file = Path.is_file
        def mock_is_file(self):
            if str(self) == "yolo26n.pt":
                return False
            return original_is_file(self)

        with patch.object(Path, "is_file", mock_is_file):
            result = resolve_yolo_model(
                "yolo26n.pt",
                allow_download=False,
                weights_dir=str(weights_dir),
            )
        assert result == str(weights_dir / "yolo26n.pt")


# ===========================================================================
# 7. Unrecognized model
# ===========================================================================

class TestUnrecognizedModel:
    def test_nonsense_alias_raises(self):
        with pytest.raises(ValueError, match="not a recognized"):
            resolve_yolo_model("totally_fake.pt")

    def test_unsupported_family_raises(self):
        with pytest.raises(ValueError, match="not a recognized"):
            resolve_yolo_model("yolov10n.pt")


# ===========================================================================
# 8. Ultralytics cache fallback (allow_download=False, no weights_dir)
# ===========================================================================

class TestUltralyticsWeightsDirFallback:
    def test_allow_download_false_finds_in_ultralytics_cache(self, tmp_path):
        """allow_download=False should check Ultralytics' configured weights_dir."""
        ul_cache = tmp_path / "ultralytics_weights"
        ul_cache.mkdir()
        (ul_cache / "yolov9t.pt").touch()

        original_is_file = Path.is_file
        def mock_is_file(self):
            if str(self) == "yolov9t.pt":
                return False
            return original_is_file(self)

        with patch.object(Path, "is_file", mock_is_file):
            with patch(
                "signdata.processors.detection.yolo.resolver._get_ultralytics_weights_dir",
                return_value=ul_cache,
            ):
                result = resolve_yolo_model("yolov9t.pt", allow_download=False)
        assert result == str(ul_cache / "yolov9t.pt")

    def test_allow_download_false_no_ultralytics_cache_raises(self):
        """allow_download=False with no cache anywhere should raise."""
        original_is_file = Path.is_file
        def mock_is_file(self):
            if self.name == "yolov9t.pt":
                return False
            return original_is_file(self)

        with patch.object(Path, "is_file", mock_is_file):
            with patch(
                "signdata.processors.detection.yolo.resolver._get_ultralytics_weights_dir",
                return_value=None,
            ):
                with pytest.raises(FileNotFoundError, match="allow_download=False"):
                    resolve_yolo_model("yolov9t.pt", allow_download=False)


# ===========================================================================
# 9. Installed Ultralytics asset support
# ===========================================================================

class TestInstalledAssetSupport:
    def test_missing_asset_catalogue_entry_raises_clear_error(self):
        with patch(
            "signdata.processors.detection.yolo.resolver._get_ultralytics_asset_stems",
            return_value={"yolov8n", "yolo11m"},
        ):
            with patch(
                "signdata.processors.detection.yolo.resolver._get_ultralytics_version",
                return_value="8.4.24",
            ):
                with pytest.raises(RuntimeError, match="does not include it in the official asset catalogue"):
                    resolve_yolo_model("yolo26n.pt")

    def test_supported_asset_catalogue_entry_passes(self):
        with patch(
            "signdata.processors.detection.yolo.resolver._get_ultralytics_asset_stems",
            return_value={"yolo26n", "yolov8n", "yolo11m"},
        ):
            result = resolve_yolo_model("yolo26n.pt")
        assert result == "yolo26n.pt"


# ===========================================================================
# 10. Schema config fields
# ===========================================================================

class TestYOLOConfigNewFields:
    def test_defaults(self):
        cfg = YOLODetectionConfig()
        assert cfg.model == "yolo11m.pt"
        assert cfg.allow_download is True
        assert cfg.weights_dir is None

    def test_custom_values(self):
        cfg = YOLODetectionConfig(
            model="yolo11m.pt",
            allow_download=False,
            weights_dir="/my/cache",
        )
        assert cfg.model == "yolo11m.pt"
        assert cfg.allow_download is False
        assert cfg.weights_dir == "/my/cache"


# ===========================================================================
# 11. Backend integration (mocked YOLO)
# ===========================================================================

class TestBackendResolverIntegration:
    def test_typo_fails_before_yolo_load(self):
        """Typo model should fail at resolver, never reaching YOLO()."""
        from signdata.processors.detection.yolo.backend import YOLODetector

        cfg = YOLODetectionConfig(model="yolov11m.pt", device="cpu")
        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
        ) as mock_yolo:
            with pytest.raises(ValueError, match="Did you mean"):
                YOLODetector(cfg)
            mock_yolo.assert_not_called()

    def test_valid_alias_passes_to_yolo(self):
        from signdata.processors.detection.yolo.backend import YOLODetector

        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cpu")
        mock_model = MagicMock()
        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
            return_value=mock_model,
        ) as mock_yolo:
            detector = YOLODetector(cfg)
            mock_yolo.assert_called_once_with("yolov8n.pt")
            mock_model.to.assert_called_once_with("cpu")

    def test_alias_file_not_found_becomes_runtime_error(self):
        """FileNotFoundError from YOLO() on a valid alias should
        surface as a download/cache RuntimeError."""
        from signdata.processors.detection.yolo.backend import YOLODetector

        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cpu")
        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
            side_effect=FileNotFoundError("no such file"),
        ):
            with pytest.raises(RuntimeError, match="download failed"):
                YOLODetector(cfg)

    def test_local_path_file_not_found_stays_file_not_found(self, tmp_path):
        """FileNotFoundError from YOLO() on a non-alias local path
        should remain a FileNotFoundError with 'weights not found'."""
        from signdata.processors.detection.yolo.backend import YOLODetector

        local_weights = tmp_path / "custom_model.pt"
        local_weights.touch()  # resolver passes (file exists)

        cfg = YOLODetectionConfig(model=str(local_weights), device="cpu")
        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
            side_effect=FileNotFoundError("corrupt file"),
        ):
            with pytest.raises(FileNotFoundError, match="YOLO weights not found"):
                YOLODetector(cfg)

    def test_yolov26_typo_fails_before_yolo_load(self):
        """yolov26n.pt typo should fail at resolver with suggestion."""
        from signdata.processors.detection.yolo.backend import YOLODetector

        cfg = YOLODetectionConfig(model="yolov26n.pt", device="cpu")
        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
        ) as mock_yolo:
            with pytest.raises(ValueError, match="Did you mean 'yolo26n.pt'"):
                YOLODetector(cfg)
            mock_yolo.assert_not_called()
