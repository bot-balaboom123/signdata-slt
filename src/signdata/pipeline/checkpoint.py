"""Stage completion markers and config-aware checkpoint validation.

Each pipeline stage writes a ``_SUCCESS.json`` marker to its output
directory on completion.  On subsequent runs the runner checks these
markers to skip stages whose inputs haven't changed.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

SUCCESS_FILENAME = "_SUCCESS.json"

# Per-stage hash field registry — new 4-stage pipeline
STAGE_HASH_FIELDS: Dict[str, List[str]] = {
    "dataset.download": [
        "dataset.source",
    ],
    "dataset.manifest": [
        "dataset.source",
    ],
    "processing": [
        "processing.processor",
        "processing.detection",
        "processing.pose",
        "processing.frame_skip",
        "processing.target_fps",
        "processing.detection_config",
        "processing.pose_config",
        "processing.video_config",
    ],
    "post_processing.normalize": [
        "post_processing.normalize.mode",
        "post_processing.normalize.remove_z",
        "post_processing.normalize.select_keypoints",
        "post_processing.normalize.keypoint_preset",
        "post_processing.normalize.keypoint_indices",
        "post_processing.normalize.mask_empty_frames",
        "post_processing.normalize.mask_low_confidence",
        "post_processing.normalize.visibility_threshold",
        "post_processing.normalize.missing_value",
    ],
    "output.webdataset": [
        "output.config",
        "processing.processor",
    ],
}


def _resolve_dotpath(obj: Any, dotpath: str) -> Any:
    """Resolve a dot-separated path on a nested object / dict."""
    current = obj
    for part in dotpath.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _stable_json(value: Any) -> str:
    """Return a deterministic JSON string for hashing."""
    return json.dumps(value, sort_keys=True, default=str)


def compute_stage_hash(
    config: Any,
    stage_name: str,
    hash_fields: Optional[List[str]] = None,
) -> str:
    """Compute a SHA-256 hash of the config fields relevant to *stage_name*."""
    fields = hash_fields or STAGE_HASH_FIELDS.get(stage_name, [])
    values = {f: _resolve_dotpath(config, f) for f in sorted(fields)}
    payload = _stable_json(values)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"sha256:{digest}"


def compute_manifest_hash(
    manifest: Union[str, Path, pd.DataFrame],
) -> str:
    """Compute a SHA-256 hash of manifest content."""
    if isinstance(manifest, pd.DataFrame):
        payload = manifest.to_csv(sep="\t", index=False).encode()
    else:
        path = Path(manifest)
        if not path.exists():
            return "sha256:missing"
        payload = path.read_bytes()

    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"


def success_content_hash(
    output_dir: Union[str, Path],
) -> str:
    """Compute a SHA-256 hash of an existing ``_SUCCESS.json`` file."""
    path = Path(output_dir) / SUCCESS_FILENAME
    if not path.exists():
        return "sha256:missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def compute_upstream_hash(
    upstream_hashes: Sequence[str],
) -> str:
    """Compute a combined hash from upstream stages' success markers."""
    combined = "\n".join(sorted(upstream_hashes))
    digest = hashlib.sha256(combined.encode()).hexdigest()
    return f"sha256:{digest}"


def write_success(
    output_dir: Union[str, Path],
    stage_name: str,
    stage_config_hash: str,
    manifest_hash: str,
    upstream_hash: str = "",
    output_count: int = 1,
    output_sample: Optional[List[str]] = None,
) -> Path:
    """Write a ``_SUCCESS.json`` marker to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = {
        "stage": stage_name,
        "stage_config_hash": stage_config_hash,
        "manifest_hash": manifest_hash,
        "upstream_success_hash": upstream_hash,
        "output_count": output_count,
        "output_sample": output_sample or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = output_dir / SUCCESS_FILENAME
    path.write_text(json.dumps(marker, indent=2))
    return path


def read_success(
    output_dir: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Read and parse a ``_SUCCESS.json`` marker."""
    path = Path(output_dir) / SUCCESS_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def check_success(
    output_dir: Union[str, Path],
    stage_config_hash: str,
    manifest_hash: str,
    upstream_hash: str = "",
) -> bool:
    """Validate an existing ``_SUCCESS.json`` against current hashes."""
    marker = read_success(output_dir)
    if marker is None:
        return False

    if marker.get("stage_config_hash") != stage_config_hash:
        return False

    if marker.get("manifest_hash") != manifest_hash:
        return False

    if upstream_hash and marker.get("upstream_success_hash") != upstream_hash:
        return False

    if marker.get("output_count", 0) <= 0:
        return False

    samples = marker.get("output_sample", [])
    if samples:
        output_dir = Path(output_dir)
        if not any((output_dir / s).exists() for s in samples):
            return False

    return True
