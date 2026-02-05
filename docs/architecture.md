# Architecture

## Entry Point

```bash
python -m sign_prep <config.yaml> [--override key=value ...]
```

`__main__.py` imports all registry modules, loads the YAML config, and hands it to `PipelineRunner`.

## Pipeline Flow

```
YAML config
  │
  ▼
load_config()          # merge _base → dataset YAML → CLI overrides
  │
  ▼
PipelineRunner(config)
  │  ├─ look up dataset in DATASET_REGISTRY
  │  └─ build processor chain from pipeline.steps
  │
  ▼
for each processor:
  │  processor.validate(context)
  │  context = processor.run(context)
  │  context.completed_steps.append(name)
  ▼
PipelineContext (final)
```

`PipelineContext` is a dataclass that carries shared state between steps: the config, dataset instance, project root, manifest path/DataFrame, completed steps, and per-step stats.

## Registry System

Three global registries in `registry.py`, populated via decorators:

| Decorator | Registry | Base class |
|---|---|---|
| `@register_dataset(name)` | `DATASET_REGISTRY` | `BaseDataset` |
| `@register_processor(name)` | `PROCESSOR_REGISTRY` | `BaseProcessor` |
| `@register_extractor(name)` | `EXTRACTOR_REGISTRY` | `LandmarkExtractor` |

Registration happens at import time. `__main__.py` imports `sign_prep.datasets`, `sign_prep.processors`, and `sign_prep.extractors` to trigger it.

## Pipeline Modes

**`pose` mode** (landmarks):
`download → manifest → extract → normalize → webdataset`
Extracts per-frame pose landmarks as `.npy` arrays, normalizes them, and packages into tar shards.

**`video` mode** (clips):
`download → manifest → clip_video → webdataset`
Clips video segments with ffmpeg and packages `.mp4` files into tar shards.

The `steps` list in config controls exactly which processors run. `start_from` and `stop_at` allow resuming or stopping partway through.

## Processors

Every processor subclasses `BaseProcessor` and implements:

- `run(context) → context` -- execute the step, return updated context
- `validate(context) → bool` -- optional pre-check (default: `True`)

Processors are stateless between runs. All shared state flows through `PipelineContext`.

## Extensibility

**New dataset:**
```python
@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    name = "my_dataset"

    @classmethod
    def validate_config(cls, config):
        ...  # dataset-specific checks
```

**New processor:**
```python
@register_processor("my_step")
class MyProcessor(BaseProcessor):
    name = "my_step"

    def run(self, context):
        ...  # processing logic
        return context
```

**New extractor:**
```python
@register_extractor("my_extractor")
class MyExtractor(LandmarkExtractor):
    def process_frame(self, frame):
        ...  # return (K, 4) array or None

    def close(self):
        ...  # release resources
```
