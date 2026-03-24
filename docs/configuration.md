# Configuration Reference

## Job and Experiment Files

Runnable job YAMLs live under `configs/jobs/<dataset>/`.
Experiment YAMLs live under `configs/experiments/` and reference job paths relative to `configs/`.

```bash
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override processing.max_workers=8 normalize.remove_z=true

python -m signdata experiment configs/experiments/baseline_youtube_asl.yaml
```

## Merge Order

Config values are applied in this order, with later values winning:

1. Pydantic defaults from `src/signdata/config/schema.py`
2. Job YAML values
3. CLI overrides passed to `python -m signdata run`
4. Per-job `overrides` from an experiment YAML

## Minimal Working Config

The smallest practical job config only needs the fields required by the
dataset adapter and recipe:

```yaml
# configs/jobs/my_dataset/mediapipe.yaml
dataset: my_dataset
recipe: pose

paths:
  root: dataset/my_dataset
  videos: dataset/my_dataset/videos
  manifest: dataset/my_dataset/manifest.csv

extractor:
  name: mediapipe
```

Some dataset adapters also require `source` fields. For example, YouTube-ASL
needs `source.video_ids_file`.

## File Layout

```text
configs/
├── experiments/
│   ├── baseline_youtube_asl.yaml
│   └── privacy_aware_slt.yaml
└── jobs/
    ├── youtube_asl/
    │   ├── mediapipe.yaml
    │   ├── mmpose.yaml
    │   └── video.yaml
    └── how2sign/
        ├── mediapipe.yaml
        ├── mmpose.yaml
        └── video.yaml
```

## Top-Level Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `dataset` | `str` | none | Dataset name registered in `DATASET_REGISTRY` |
| `recipe` | `"pose"` \| `"video"` | `"pose"` | Recipe that defines legal stage order |
| `run_name` | `str` | `"default"` | Output namespace under dataset artifacts |
| `start_from` | `str?` | `null` | Start from this stage |
| `stop_at` | `str?` | `null` | Stop after this stage |
| `source` | `dict` | `{}` | Dataset-specific source/acquire/manifest settings |
| `stage_config` | `dict` | `{}` | Optional per-stage settings keyed by stage name |

## `paths`

All relative paths are resolved from the project root.

| Field | Type | Default | Description |
|---|---|---|---|
| `root` | `str` | `""` | Dataset root directory |
| `videos` | `str` | `""` | Source videos |
| `transcripts` | `str` | `""` | Transcript JSON files |
| `manifest` | `str` | `""` | Base manifest CSV |
| `landmarks` | `str` | `""` | Raw extracted landmarks |
| `normalized` | `str` | `""` | Normalized landmarks |
| `clips` | `str` | `""` | Clipped videos (used by `clip_video`) |
| `webdataset` | `str` | `""` | Output shard directory |

## `source`

`source` is dataset-specific. Common keys used by the built-in datasets are:

| Field | Type | Default | Description |
|---|---|---|---|
| `video_ids_file` | `str` | `""` | Video ID list for YouTube-style datasets |
| `languages` | `list[str]` | `["en"]` | Transcript language codes |
| `download_format` | `str` | `"worstvideo[height>=720]..."` | yt-dlp format selector |
| `rate_limit` | `str` | `"5M"` | Download rate limit |
| `concurrent_fragments` | `int` | `5` | Parallel download fragments |
| `max_text_length` | `int` | `300` | Max caption length |
| `min_duration` | `float` | `0.2` | Min segment duration |
| `max_duration` | `float` | `60.0` | Max segment duration |
| `manifest_csv` | `str` | `""` | Existing manifest path for datasets such as How2Sign |
| `split` | `str` | `"all"` | Split label for datasets such as How2Sign |

## `extractor`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"mediapipe"` | Extractor name |
| `max_workers` | `int` | `4` | Parallel extraction workers |
| `batch_size` | `int` | `16` | Frames per inference batch |
| `model_complexity` | `int` | `1` | MediaPipe model complexity |
| `min_detection_confidence` | `float` | `0.5` | MediaPipe detection threshold |
| `min_tracking_confidence` | `float` | `0.5` | MediaPipe tracking threshold |
| `refine_face_landmarks` | `bool` | `true` | Use refined face landmarks |
| `pose_model_config` | `str` | `""` | MMPose model config path |
| `pose_model_checkpoint` | `str` | `""` | MMPose checkpoint path |
| `det_model_config` | `str` | `""` | Detector config path |
| `det_model_checkpoint` | `str` | `""` | Detector checkpoint path |
| `bbox_threshold` | `float` | `0.5` | MMPose person-box threshold |
| `keypoint_threshold` | `float` | `0.3` | MMPose keypoint threshold |
| `add_visible` | `bool` | `true` | Keep visibility in output |
| `device` | `str` | `"cuda:0"` | Inference device |

## `normalize`

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `"xy_isotropic_z_minmax"` | Normalization mode |
| `remove_z` | `bool` | `false` | Drop z after normalization |
| `select_keypoints` | `bool` | `true` | Reduce keypoints before flattening |
| `keypoint_preset` | `str?` | `null` | Named preset from `signdata.pose.presets` |
| `keypoint_indices` | `list[int]?` | `null` | Explicit keypoint indices |
| `mask_empty_frames` | `bool` | `true` | Mask all-zero frames |
| `mask_low_confidence` | `bool` | `false` | Mask low-visibility landmarks |
| `visibility_threshold` | `float` | `0.3` | Visibility cutoff |
| `missing_value` | `float` | `-999.0` | Fill value for masked data |

## `processing`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_workers` | `int` | `4` | Generic worker count |
| `target_fps` | `float?` | `24.0` | Target sampling FPS |
| `frame_skip` | `int` | `2` | Skip every N frames |
| `accept_fps_range` | `list[float]?` | `[24.0, 60.0]` | Allowed input FPS range |
| `skip_existing` | `bool` | `true` | Skip existing outputs |
| `min_duration` | `float` | `0.2` | Min segment duration during extraction |
| `max_duration` | `float` | `60.0` | Max segment duration during extraction |
| `signer_policy` | `str` | `"primary_signer"` | Multi-person handling policy |

## `clip_video`

| Field | Type | Default | Description |
|---|---|---|---|
| `codec` | `str` | `"copy"` | ffmpeg codec for clipped videos |
| `resize` | `list[int]?` | `null` | Optional `[width, height]` resize |

## `webdataset`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_shard_count` | `int` | `10000` | Max samples per shard |
| `max_shard_size` | `int?` | `null` | Max shard size in bytes |

## `stage_config`

`stage_config` holds settings for optional stages such as `obfuscate` and
`window_video`.

```yaml
stage_config:
  obfuscate:
    method: blur

  window_video:
    window_seconds: 10.0
    stride_seconds: 5.0
    min_window_seconds: 2.0
```

## CLI Examples

```bash
# Change number of workers
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override processing.max_workers=8

# Switch extractor
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override extractor.name=mmpose

# Run only part of the recipe
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --from extract --to normalize

# Override run name for isolated outputs
python -m signdata run configs/jobs/youtube_asl/video.yaml \
  --run-name privacy_v1
```

## See Also

- [Architecture](architecture.md) -- runner, registry, and recipes
- [Pipeline Stages](pipeline-stages.md) -- stage behavior and outputs
- [Installation Guide](installation.md) -- environment setup and MMPose dependencies
