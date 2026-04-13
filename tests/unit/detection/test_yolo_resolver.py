"""Tests for YOLO model resolver: alias validation, typo suggestions, cache logic."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from signdata.processors.detection.yolo.resolver import (
    SUPPORTED_FAMILIES,
    VALID_ALIASES,
    is_valid_alias,
    resolve_yolo_model,
)
from signdata.processors.detection.yolo.backend import YOLODetector
from signdata.config.schema import YOLODetectionConfig


@pytest.fixture
def block_is_file():
    """Patch Path.is_file so paths whose str matches any given value return False.

    Useful for simulating "alias not found in cwd" without blocking cached
    copies under tmp_path, which have a different str representation.
    """
    def _block(*paths: str):
        blocked = set(paths)
        original = Path.is_file

        def fake(self):
            if str(self) in blocked:
                return False
            return original(self)

        return patch.object(Path, "is_file", fake)

    return _block


# ===========================================================================
# 1. Alias catalogue
# ===========================================================================

class TestAliasCatalogue:
    @pytest.mark.parametrize(
        "family,sizes",
        list(SUPPORTED_FAMILIES.items()),
        ids=list(SUPPORTED_FAMILIES.keys()),
    )
    def test_family_aliases_present(self, family, sizes):
        for size in sizes:
            assert f"{family}{size}.pt" in VALID_ALIASES

    @pytest.mark.parametrize(
        "alias",
        ["yolov10n.pt", "yolov11m.pt", "yolov26n.pt"],
    )
    def test_invalid_alias_not_in_set(self, alias):
        assert alias not in VALID_ALIASES

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("yolov8n.pt", True),
            ("yolo11m.pt", True),
            ("yolo26n.pt", True),
            ("yolov11m.pt", False),
            ("random.pt", False),
            ("yolov8n", True),
            ("yolo11m", True),
            ("yolo26n", True),
            ("yolov26n", False),
        ],
    )
    def test_is_valid_alias(self, model, expected):
        assert is_valid_alias(model) is expected


# ===========================================================================
# 2. Local file resolution
# ===========================================================================

class TestLocalFileResolution:
    def test_existing_local_file(self, tmp_path):
        weights = tmp_path / "my_custom_model.pt"
        weights.touch()
        assert resolve_yolo_model(str(weights)) == str(weights)

    def test_cached_in_weights_dir(self, tmp_path, block_is_file):
        weights_dir = tmp_path / "cache"
        weights_dir.mkdir()
        (weights_dir / "yolo11n.pt").touch()

        with block_is_file("yolo11n.pt"):
            result = resolve_yolo_model("yolo11n.pt", weights_dir=str(weights_dir))
        assert result == str(weights_dir / "yolo11n.pt")


# ===========================================================================
# 3. Typo detection
# ===========================================================================

class TestTypoDetection:
    @pytest.mark.parametrize(
        "typo,suggestion",
        [
            ("yolov11m.pt", "yolo11m.pt"),
            ("yolov11n.pt", "yolo11n.pt"),
            ("yolov11x.pt", "yolo11x.pt"),
            ("yolov26n.pt", "yolo26n.pt"),
            ("yolov26m.pt", "yolo26m.pt"),
            ("yolov26n", "yolo26n.pt"),
        ],
    )
    def test_typo_suggests_correction(self, typo, suggestion):
        with pytest.raises(ValueError, match=f"Did you mean '{suggestion}'"):
            resolve_yolo_model(typo)


# ===========================================================================
# 4. Valid alias with download enabled
# ===========================================================================

class TestValidAliasDownload:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("yolov8n.pt", "yolov8n.pt"),
            ("yolo11m.pt", "yolo11m.pt"),
            ("yolov9c.pt", "yolov9c.pt"),
            ("yolo26n.pt", "yolo26n.pt"),
            ("yolov8n", "yolov8n.pt"),
            ("yolo26n", "yolo26n.pt"),
        ],
    )
    def test_alias_returned_normalised(self, model, expected):
        assert resolve_yolo_model(model, allow_download=True) == expected


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
    def test_alias_no_cache_raises(self, block_is_file):
        with block_is_file("yolo11x.pt"):
            with pytest.raises(FileNotFoundError, match="allow_download=False"):
                resolve_yolo_model(
                    "yolo11x.pt",
                    allow_download=False,
                    weights_dir="/nonexistent/path",
                )

    @pytest.mark.parametrize("alias", ["yolo11m.pt", "yolo26n.pt"])
    def test_alias_cached_in_weights_dir(self, tmp_path, block_is_file, alias):
        weights_dir = tmp_path / "cache"
        weights_dir.mkdir()
        (weights_dir / alias).touch()

        with block_is_file(alias):
            result = resolve_yolo_model(
                alias,
                allow_download=False,
                weights_dir=str(weights_dir),
            )
        assert result == str(weights_dir / alias)


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
    def test_allow_download_false_finds_in_ultralytics_cache(self, tmp_path, block_is_file):
        ul_cache = tmp_path / "ultralytics_weights"
        ul_cache.mkdir()
        (ul_cache / "yolov9t.pt").touch()

        with block_is_file("yolov9t.pt"), patch(
            "signdata.processors.detection.yolo.resolver._get_ultralytics_weights_dir",
            return_value=ul_cache,
        ):
            result = resolve_yolo_model("yolov9t.pt", allow_download=False)
        assert result == str(ul_cache / "yolov9t.pt")

    def test_allow_download_false_no_ultralytics_cache_raises(self, block_is_file):
        with block_is_file("yolov9t.pt"), patch(
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
        assert cfg.model == "yolov8n.pt"
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
    @pytest.mark.parametrize(
        "typo,expected_match",
        [
            ("yolov11m.pt", "Did you mean"),
            ("yolov26n.pt", "Did you mean 'yolo26n.pt'"),
        ],
    )
    def test_typo_fails_before_yolo_load(self, typo, expected_match):
        cfg = YOLODetectionConfig(model=typo, device="cpu")
        with patch("signdata.processors.detection.yolo.backend.YOLO") as mock_yolo:
            with pytest.raises(ValueError, match=expected_match):
                YOLODetector(cfg)
            mock_yolo.assert_not_called()

    def test_valid_alias_passes_to_yolo(self):
        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cpu")
        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
            return_value=MagicMock(),
        ) as mock_yolo:
            YOLODetector(cfg)
            mock_yolo.assert_called_once_with("yolov8n.pt")

    def test_alias_file_not_found_becomes_runtime_error(self):
        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cpu")
        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
            side_effect=FileNotFoundError("no such file"),
        ):
            with pytest.raises(RuntimeError, match="download failed"):
                YOLODetector(cfg)

    def test_local_path_file_not_found_stays_file_not_found(self, tmp_path):
        local_weights = tmp_path / "custom_model.pt"
        local_weights.touch()

        cfg = YOLODetectionConfig(model=str(local_weights), device="cpu")
        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
            side_effect=FileNotFoundError("corrupt file"),
        ):
            with pytest.raises(FileNotFoundError, match="YOLO weights not found"):
                YOLODetector(cfg)
