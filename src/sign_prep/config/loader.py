"""YAML config loading with dataset defaults merging."""

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .schema import Config
from ..registry import DATASET_REGISTRY


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_paths(config: Config, project_root: Path) -> Config:
    """Resolve relative paths to absolute paths based on project root."""
    paths = config.paths

    if not paths.root:
        paths.root = str(project_root / "dataset" / config.dataset)

    root = Path(paths.root)
    if not root.is_absolute():
        root = project_root / root
        paths.root = str(root)

    extractor_name = config.extractor.name

    if not paths.videos:
        paths.videos = str(root / "videos")
    elif not Path(paths.videos).is_absolute():
        paths.videos = str(project_root / paths.videos)

    if not paths.transcripts:
        paths.transcripts = str(root / "transcripts")
    elif not Path(paths.transcripts).is_absolute():
        paths.transcripts = str(project_root / paths.transcripts)

    if not paths.manifest:
        paths.manifest = str(root / "manifest.csv")
    elif not Path(paths.manifest).is_absolute():
        paths.manifest = str(project_root / paths.manifest)

    if not paths.landmarks:
        paths.landmarks = str(root / "landmarks" / extractor_name)
    elif not Path(paths.landmarks).is_absolute():
        paths.landmarks = str(project_root / paths.landmarks)

    if not paths.normalized:
        paths.normalized = str(root / "normalized" / extractor_name)
    elif not Path(paths.normalized).is_absolute():
        paths.normalized = str(project_root / paths.normalized)

    if not paths.clips:
        paths.clips = str(root / "clips")
    elif not Path(paths.clips).is_absolute():
        paths.clips = str(project_root / paths.clips)

    if not paths.webdataset:
        paths.webdataset = str(
            root / "webdataset" / config.pipeline.mode / extractor_name
        )
    elif not Path(paths.webdataset).is_absolute():
        paths.webdataset = str(project_root / paths.webdataset)

    # Resolve download.video_ids_file relative to project root
    vid_file = config.download.video_ids_file
    if vid_file and not Path(vid_file).is_absolute():
        config.download.video_ids_file = str(project_root / vid_file)

    return config


def load_config(
    yaml_path: str,
    overrides: Optional[List[str]] = None,
) -> Config:
    """Load config from YAML file, merging with dataset defaults.

    Merge order (later overrides earlier):
    1. Pydantic defaults (hardcoded in schema)
    2. Dataset defaults (from BaseDataset.default_config())
    3. YAML config file
    4. CLI overrides (key=value pairs)
    """
    yaml_path = os.path.abspath(yaml_path)
    project_root = Path(yaml_path).parent.parent
    if project_root.name == "configs":
        project_root = project_root.parent

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    dataset_name = raw.get("dataset")
    if not dataset_name:
        raise ValueError("Config must specify 'dataset' field")

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    # Get dataset defaults
    dataset_cls = DATASET_REGISTRY[dataset_name]
    defaults = dataset_cls.default_config()

    # Merge: defaults <- yaml
    merged = deep_merge(defaults, raw)

    # Apply CLI overrides
    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Override must be key=value, got: {override}")
            key, value = override.split("=", 1)
            _set_nested(merged, key, _parse_value(value))

    # Populate pipeline steps from dataset if not specified
    pipeline_cfg = merged.get("pipeline", {})
    mode = pipeline_cfg.get("mode", "pose")
    if not pipeline_cfg.get("steps"):
        steps = dataset_cls.pipeline_steps(mode)
        if "pipeline" not in merged:
            merged["pipeline"] = {}
        merged["pipeline"]["steps"] = steps

    config = Config(**merged)
    config = resolve_paths(config, project_root)
    return config


def _set_nested(d: Dict, key: str, value: Any) -> None:
    """Set a nested dictionary value using dot-separated key."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _parse_value(value: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none" or value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
