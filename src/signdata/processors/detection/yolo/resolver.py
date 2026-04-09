"""YOLO model resolver: validates aliases, suggests typo fixes, manages cache."""

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported detect-only family rules: family prefix -> allowed size suffixes.
# Families whose official name drops the "v" (yolo11, yolo26) are listed
# without "v"; families that keep it (yolov8, yolov9) include "v".
# ---------------------------------------------------------------------------

SUPPORTED_FAMILIES: dict[str, list[str]] = {
    "yolov8": ["n", "s", "m", "l", "x"],
    "yolov9": ["t", "s", "m", "c", "e"],
    "yolo11": ["n", "s", "m", "l", "x"],
    "yolo26": ["n", "s", "m", "l", "x"],
}

# Build the full set of valid aliases in normalised .pt form.
VALID_ALIASES: set[str] = set()
for _family, _sizes in SUPPORTED_FAMILIES.items():
    for _size in _sizes:
        VALID_ALIASES.add(f"{_family}{_size}.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_stem(model: str) -> str:
    """Ensure model string ends with ``.pt`` (Ultralytics accepts bare stems
    but always normalizes to ``.pt`` internally)."""
    if not model.endswith(".pt"):
        return model + ".pt"
    return model


def _is_hub_model(model: str) -> bool:
    """Return True for Ultralytics HUB model URLs."""
    try:
        from ultralytics.hub import HUB_WEB_ROOT
        hub_root = HUB_WEB_ROOT
    except Exception:
        hub_root = "https://hub.ultralytics.com"
    return model.startswith(f"{hub_root}/models/")


def _is_remote_weights_url(model: str) -> bool:
    """Return True for remote weight URLs that Ultralytics downloads locally."""
    url = urlsplit(model)
    if url.scheme not in {"http", "https", "ul"}:
        return False
    return url.path.endswith((".pt", ".pth"))


def _is_triton_model(model: str) -> bool:
    """Return True for Triton model URLs that Ultralytics routes directly."""
    url = urlsplit(model)
    return bool(url.netloc and url.path and url.scheme in {"http", "grpc"})


def _suggest_correction(model: str) -> Optional[str]:
    """Return a suggested correction for a mistyped alias, or None.

    Catches the common mistake of writing ``yolov11*`` or ``yolov26*``
    instead of the official ``yolo11*`` / ``yolo26*``.
    """
    # Pattern: yolov<digits><size>.pt  where the no-v form is official
    m = re.match(r"^yolov(\d+)([a-z])\.pt$", model)
    if m:
        candidate = f"yolo{m.group(1)}{m.group(2)}.pt"
        if candidate in VALID_ALIASES:
            family = f"yolo{m.group(1)}"
            return candidate
    return None


def _get_ultralytics_weights_dir() -> Optional[Path]:
    """Read (not write) the current Ultralytics weights directory setting."""
    try:
        from ultralytics import settings
        raw = settings.get("weights_dir", None)
        if raw:
            return Path(str(raw))
    except Exception:
        pass
    return None


def _get_ultralytics_asset_stems() -> Optional[set[str]]:
    """Return the installed Ultralytics asset stem catalogue if available."""
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
        return set(GITHUB_ASSETS_STEMS)
    except Exception:
        return None


def _get_ultralytics_version() -> Optional[str]:
    """Return the installed ultralytics version string, or None."""
    try:
        from ultralytics import __version__
        return __version__
    except Exception:
        return None


def _family_of(alias: str) -> Optional[str]:
    """Extract the family prefix from a normalised alias like 'yolo26n.pt'."""
    m = re.match(r"^(yolov?\d+)[a-z]\.pt$", alias)
    return m.group(1) if m else None


def _check_installed_alias_support(alias: str) -> None:
    """Raise RuntimeError if the installed Ultralytics package lacks this alias."""
    asset_stems = _get_ultralytics_asset_stems()
    if asset_stems is None:
        return

    stem = alias.removesuffix(".pt")
    if stem not in asset_stems:
        installed = _get_ultralytics_version() or "unknown"
        raise RuntimeError(
            f"YOLO model {alias!r} is an official alias for this backend, "
            f"but the installed ultralytics {installed} package does not "
            "include it in the official asset catalogue. Upgrade ultralytics "
            "to a version that supports this model family."
        )


def is_valid_alias(model: str) -> bool:
    """Check if model string is a recognized Ultralytics detect alias.

    Accepts both bare stems (``yolo11m``) and ``.pt`` form (``yolo11m.pt``).
    """
    return _normalize_stem(model) in VALID_ALIASES


def resolve_yolo_model(
    model: str,
    *,
    allow_download: bool = True,
    weights_dir: Optional[str] = None,
) -> str:
    """Resolve a YOLO model string to a loadable path or alias.

    Args:
        model: Local file path or Ultralytics alias (e.g. ``yolo11m.pt``
            or bare stem ``yolo11m``).
        allow_download: If False, only accept existing local files.
        weights_dir: Extra directory to check for cached weights.
            This is a read-only search path — it does not mutate
            Ultralytics' global settings.

    Returns:
        The resolved model string (always ``.pt`` normalised) to pass to
        ``YOLO()``.

    Raises:
        FileNotFoundError: Model file not found locally when download is
            disabled.
        ValueError: Model alias is not supported or appears to be a typo.
        RuntimeError: Model alias is not supported by the installed
            Ultralytics package.
    """
    model = model.strip()
    local_path = Path(model)

    # --- 1. Existing local file (absolute or relative) ---
    if local_path.is_file():
        logger.info("YOLO model resolved to local file: %s", local_path)
        return str(local_path)

    # --- 2. Remote / upstream-managed model references ---
    if _is_hub_model(model):
        if not allow_download:
            raise FileNotFoundError(
                f"YOLO model {model!r} is a remote Ultralytics HUB reference. "
                "Set allow_download=True or provide a local weights path."
            )
        return model

    if _is_triton_model(model):
        # Triton endpoints are remote services, not downloadable weights.
        return model

    if _is_remote_weights_url(model):
        if not allow_download:
            raise FileNotFoundError(
                f"YOLO model {model!r} is a remote weights URL. "
                "Set allow_download=True or provide a local weights path."
            )
        return model

    # Normalise bare stems to .pt for all remaining alias/cache checks.
    normalised = _normalize_stem(model)

    # --- 3. Check inside weights_dir if provided ---
    if weights_dir:
        cached = Path(weights_dir) / normalised
        if cached.is_file():
            logger.info("YOLO model resolved from cache: %s", cached)
            return str(cached)

    # --- 4. Typo detection (before alias validation) ---
    suggestion = _suggest_correction(normalised)
    if suggestion:
        family = _family_of(suggestion)
        raise ValueError(
            f"Unknown YOLO model {model!r}. Did you mean {suggestion!r}? "
            f"{family.upper()} uses the naming scheme "
            f"'{family}<size>.pt', not '{family.replace('yolo', 'yolov')}<size>.pt'."
        )

    # --- 5. Valid alias ---
    if normalised in VALID_ALIASES:
        _check_installed_alias_support(normalised)

        if not allow_download:
            # Check Ultralytics' configured weights dir (read-only)
            ul_dir = _get_ultralytics_weights_dir()
            if ul_dir:
                candidate = ul_dir / normalised
                if candidate.is_file():
                    logger.info(
                        "YOLO model found in Ultralytics cache "
                        "(download disabled): %s",
                        candidate,
                    )
                    return str(candidate)

            raise FileNotFoundError(
                f"YOLO model {normalised!r} is a valid alias but "
                f"allow_download=False and no cached weights found. Either "
                f"set allow_download=True or provide a local path."
            )

        logger.info(
            "YOLO model %r is a valid alias; Ultralytics will download if needed.",
            normalised,
            )
        return normalised

    # --- 6. Not a local file and not a recognized alias ---
    raise ValueError(
        f"YOLO model {model!r} is not an existing file and not a recognized "
        f"Ultralytics detect alias. Supported aliases: "
        f"{', '.join(sorted(VALID_ALIASES))}"
    )


__all__ = [
    "SUPPORTED_FAMILIES",
    "VALID_ALIASES",
    "is_valid_alias",
    "resolve_yolo_model",
]
