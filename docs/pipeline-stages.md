# Pipeline Stages

Stage order is recipe-driven. The runner starts from `config.recipe` and then
applies `start_from` / `stop_at` if present.

- `pose` recipe: `acquire → manifest → detect_person → window_video → clip_video → crop_video → extract → normalize → webdataset`
- `video` recipe: `acquire → manifest → detect_person → window_video → clip_video → crop_video → obfuscate → webdataset`

Optional stages only run when enabled by config or manifest data.

## Stage Summary

| Stage | Recipe(s) | Output | Activation |
|---|---|---|---|
| `acquire` | both | videos and transcripts, or validation of local data | always |
| `manifest` | both | base manifest CSV | always |
| `detect_person` | both | derived manifest with `BBOX_*` columns | `detect_person.enabled: true` |
| `window_video` | both | derived manifest with windowed rows | `stage_config.window_video` |
| `clip_video` | both | sentence or window clips in `paths.clips` | manifest has timing |
| `crop_video` | both | cropped clips in `paths.cropped_clips` | `crop_video.enabled: true` |
| `extract` | pose | landmark arrays in `paths.landmarks` | always in pose recipe |
| `normalize` | pose | normalized arrays in `paths.normalized` | always in pose recipe |
| `obfuscate` | video | obfuscated videos in `{root}/obfuscated/{run_name}` | `stage_config.obfuscate` |
| `webdataset` | both | shards in `paths.webdataset` | always |

## `acquire`

Handles dataset acquisition.

- YouTube-ASL reads `source.video_ids_file`, downloads videos into `paths.videos`, and downloads transcript JSON into `paths.transcripts`.
- How2Sign does not download data; it validates that local inputs exist.

Key config paths:

- `source.video_ids_file`
- `source.download_format`
- `source.languages`
- `source.rate_limit`
- `source.concurrent_fragments`

## `manifest`

Builds or loads the base manifest.

- YouTube-ASL parses transcript JSON into a manifest.
- How2Sign loads an existing CSV from `paths.manifest` or `source.manifest_csv`.

Key config paths:

- `source.max_text_length`
- `source.min_duration`
- `source.max_duration`
- `paths.manifest`

## `detect_person`

Runs person detection on sampled frames and writes a derived manifest at:

```text
{paths.root}/detect_person/{run_name}/manifest.csv
```

New manifest columns:

- `BBOX_X1`, `BBOX_Y1`, `BBOX_X2`, `BBOX_Y2`
- `PERSON_DETECTED`

## `window_video`

Creates a metadata-only derived manifest at:

```text
{paths.root}/window_video/{run_name}/manifest.csv
```

It does not create video files. Downstream `clip_video` uses the windowed
manifest.

## `clip_video`

Creates physical clips in:

```text
{paths.clips}/{sample_id}.mp4
```

Key config paths:

- `clip_video.codec`
- `clip_video.resize`

## `crop_video`

Reads clips plus `BBOX_*` columns from the current manifest and writes cropped
clips into:

```text
{paths.cropped_clips}/{sample_id}.mp4
```

Key config paths:

- `crop_video.enabled`
- `crop_video.padding`
- `crop_video.codec`

## `extract`

Reads source videos or clips from the current routed `video_dir` and writes
landmarks into:

```text
{paths.landmarks}/{sample_id}.npy
```

Key config paths:

- `extractor.name`
- `extractor.pose_model_config`
- `extractor.det_model_config`
- `processing.target_fps`
- `processing.frame_skip`

## `normalize`

Reads landmark arrays from `paths.landmarks` and writes normalized arrays into:

```text
{paths.normalized}/{sample_id}.npy
```

Key config paths:

- `normalize.keypoint_preset`
- `normalize.keypoint_indices`
- `normalize.mode`
- `normalize.remove_z`
- `normalize.missing_value`

## `obfuscate`

Reads videos from the current routed `video_dir` and writes obfuscated outputs
into:

```text
{paths.root}/obfuscated/{run_name}/{file_id}.mp4
```

Configured through `stage_config.obfuscate`.

## `webdataset`

Packages the current manifest plus the active artifact directory into shards:

```text
{paths.webdataset}/shard-000000.tar
```

- Pose jobs package normalized `.npy` files.
- Video jobs package the current `.mp4` files.

## See Also

- [Architecture](architecture.md) -- runner, recipes, and routing
- [Configuration Reference](configuration.md) -- config layout and fields
- [Datasets](datasets.md) -- dataset-specific setup
