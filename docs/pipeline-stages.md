# Pipeline Stages

The pipeline runner dispatches to the processor specified by
`config.processing.processor`. Supported processors:

- `video2pose` — video → pose landmarks (.npy)
- `video2crop` — video → cropped video (.mp4)

Additional registered stages (`clip_video`, `window_video`, `obfuscate`) can be
used standalone but are not part of the main processing dispatch.

## Stage Summary

| Stage | Output | Activation |
|---|---|---|
| `acquire` | videos and transcripts, or validation of local data | always |
| `manifest` | base manifest CSV | always |
| `window_video` | derived manifest with windowed rows | `stage_config.window_video` |
| `clip_video` | sentence or window clips in `paths.clips` | manifest has timing |
| `obfuscate` | obfuscated videos in `{root}/obfuscated/{run_name}` | `stage_config.obfuscate` |
| `webdataset` | shards in `paths.webdataset` | always |

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

- [Architecture](architecture.md) -- runner and routing
- [Configuration Reference](configuration.md) -- config layout and fields
- [Datasets](datasets.md) -- dataset-specific setup
