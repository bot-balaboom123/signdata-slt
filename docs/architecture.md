# Architecture

## Entry Points

```bash
python -m signdata run <job.yaml> [--override key=value ...]
python -m signdata experiment <experiment.yaml>
```

`src/signdata/__main__.py` imports `signdata.datasets`,
`signdata.processors`, and the pose backend subpackages to populate the
registries before any config is executed.

## Run Flow

```text
job YAML
  │
  ▼
load_config()          # resolve paths and apply overrides
  │
  ▼
PipelineRunner(config)
  │  ├─ look up dataset in DATASET_REGISTRY
  │  ├─ derive stage list from config.recipe
  │  └─ slice with start_from / stop_at if requested
  │
  ▼
for each active stage:
  │  dataset adapter handles acquire / manifest
  │  registered processor handles other stages
  │  context routing is updated after each stage
  ▼
PipelineContext (final)
```

Experiment runs load `configs/experiments/*.yaml`, then execute each referenced
job under `configs/jobs/...` with its own override set.

## Registries

Three global registries live in `src/signdata/registry.py`:

| Decorator | Registry | Base class |
|---|---|---|
| `@register_dataset(name)` | `DATASET_REGISTRY` | `BaseDataset` |
| `@register_processor(name)` | `PROCESSOR_REGISTRY` | `BaseProcessor` |
| `@register_extractor(name)` | `EXTRACTOR_REGISTRY` | `LandmarkExtractor` |

## Recipes

Recipes live in `src/signdata/pipeline/recipes.py`.

- `pose`: `acquire → manifest → detect_person → window_video → clip_video → crop_video → extract → normalize → webdataset`
- `video`: `acquire → manifest → detect_person → window_video → clip_video → crop_video → obfuscate → webdataset`

Some stages are optional and only run when enabled by config or manifest data.
Examples:

- `detect_person` requires `detect_person.enabled: true`
- `crop_video` requires `crop_video.enabled: true`
- `obfuscate` requires `stage_config.obfuscate`
- `window_video` requires `stage_config.window_video`

## PipelineContext

`PipelineContext` carries shared state between stages:

| Field | Type | Description |
|---|---|---|
| `config` | `Config` | Full parsed config |
| `dataset` | `DatasetAdapter` | Active dataset adapter |
| `project_root` | `Path` | Repository root |
| `manifest_path` | `Path?` | Current manifest path |
| `manifest_df` | `DataFrame?` | Loaded manifest |
| `video_dir` | `Path?` | Current video source directory |
| `stage_output_dir` | `Path?` | Output directory for the active stage |
| `completed_steps` | `list[str]` | Completed stage names |
| `stats` | `dict[str, dict]` | Per-stage counters |

The runner updates routing fields such as `manifest_path` and `video_dir` after
each stage so downstream stages read the correct artifacts instead of
hardcoding earlier paths.

## Package Layout

- `src/signdata/datasets/` contains dataset adapters.
- `src/signdata/pose/` contains extractors and MMPose variants.
- `src/signdata/detection/` contains detector backends.
- `src/signdata/processors/` contains stage implementations grouped by domain.
- `resources/` contains shipped model config assets.

## See Also

- [Configuration Reference](configuration.md) -- config layout and key fields
- [Pipeline Stages](pipeline-stages.md) -- stage-by-stage behavior
- [Datasets](datasets.md) -- dataset-specific setup notes
