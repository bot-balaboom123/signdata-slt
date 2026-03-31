# Architecture

## Entry Points

```bash
python -m signdata run <job.yaml> [--override key=value ...]
python -m signdata experiment <experiment.yaml>
```

`src/signdata/__main__.py` imports `signdata.datasets`,
`signdata.processors`, `signdata.post_processors`, and `signdata.output`
to populate registries before any config is executed.

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
  │  ├─ resolve run-scoped output paths
  │  ├─ run dataset.download if enabled
  │  ├─ run dataset.manifest if enabled
  │  ├─ run processing.<processor> if enabled
  │  ├─ run post_processing.<recipe> entries if enabled
  │  └─ run output.<type> if enabled
  │
  ▼
PipelineContext (final)
```

Experiment runs load `configs/experiments/*.yaml`, then execute each referenced
job under `configs/jobs/...` with its own override set.

## Registries

Four global registries live in `src/signdata/registry.py`:

| Decorator | Registry | Base class |
|---|---|---|
| `@register_dataset(name)` | `DATASET_REGISTRY` | `DatasetAdapter` |
| `@register_processor(name)` | `PROCESSOR_REGISTRY` | `BaseProcessor` |
| `@register_post_processor(name)` | `POST_PROCESSOR_REGISTRY` | `BasePostProcessor` |
| `@register_output(name)` | `OUTPUT_REGISTRY` | `BaseOutput` |

## Processing

The pipeline runner dispatches to the processor specified by
`config.processing.processor`:

- `video2pose` — video → pose landmarks (.npy), using detection + pose backends
- `video2crop` — video → cropped video (.mp4), using detection + `src/signdata/processors/video/ffmpeg.py`

Post-processing then runs entries from `config.post_processing.recipes`. The
built-in recipe is `normalize`.

## PipelineContext

`PipelineContext` carries shared state between stages:

| Field | Type | Description |
|---|---|---|
| `config` | `Config` | Full parsed config |
| `dataset` | `DatasetAdapter` | Active dataset adapter |
| `output_dir` | `Path?` | Run-scoped output directory: `{paths.output}/{run_name}` |
| `webdataset_dir` | `Path?` | Run-scoped shard directory: `{paths.webdataset}/{run_name}` |
| `videos_dir` | `Path?` | Source video directory |
| `manifest_path` | `Path?` | Current manifest path |
| `manifest_df` | `DataFrame?` | Loaded manifest |
| `force_all` | `bool` | Rerun outputs even if files already exist |
| `completed_stages` | `list[str]` | Completed stage names |
| `stats` | `dict[str, dict]` | Per-stage counters |

The runner resolves run-scoped output paths once at startup, then each stage
reads and writes through `PipelineContext` instead of hardcoding artifact paths.

## Package Layout

- `src/signdata/config/` contains Pydantic config schemas (`schema.py`), YAML loading and path resolution (`loader.py`), and experiment config parsing (`experiment.py`).
- `src/signdata/pipeline/` contains the pipeline runner (`runner.py`), shared pipeline context (`context.py`), stage checkpoint/success markers (`checkpoint.py`), and the experiment runner (`experiment.py`).
- `src/signdata/datasets/` contains dataset adapter packages.
- `src/signdata/datasets/_shared/` contains dataset-ingestion helpers used only during `dataset.download` and `dataset.manifest`.
- `src/signdata/processors/detection/` contains detector backends and bbox utilities.
- `src/signdata/processors/pose/` contains pose estimators and presets.
- `src/signdata/processors/sampler/` contains frame sampling utilities for `video2pose`.
- `src/signdata/processors/video/` contains shared video helpers such as `ffmpeg.py`.
- `src/signdata/processors/` contains top-level processors such as `video2pose` and `video2crop`.
- `src/signdata/post_processors/` contains post-processing recipes such as `normalize`.
- `src/signdata/output/` contains output writers such as `webdataset`.
- `src/signdata/utils/` contains pipeline-wide helpers for video I/O, file discovery, text normalization, manifest reading/validation, and other generic logic.
- `resources/` contains shipped model config assets.

## Dataset Adapter Structure

All datasets must be packages now. The default structure is:

```text
src/signdata/datasets/
├── _shared/
│   ├── availability.py
│   ├── classmap.py
│   ├── media.py
│   ├── paths.py
│   └── youtube.py
└── <dataset_name>/
    ├── __init__.py
    ├── adapter.py
    ├── source.py
    └── manifest.py
```

- `adapter.py` registers the dataset and keeps stage methods thin.
- `source.py` owns `SourceConfig`, path resolution, validation, download, and preparation/materialization.
- `manifest.py` owns source parsing and canonical manifest construction.

Add more files only when the dataset is large enough to justify them. Typical
extensions are `schema.py`, `constants.py`, `parsing.py`, or `splits.py`.

## See Also

- [Configuration Reference](configuration.md) -- config layout and key fields
- [Pipeline Stages](pipeline-stages.md) -- stage-by-stage behavior
- [Datasets](datasets.md) -- dataset-specific setup notes
