# Contributing

## Project Structure

```text
signdata/
├── configs/
│   ├── experiments/            # Multi-job experiment configs
│   │   ├── baseline_youtube_asl.yaml
│   │   └── privacy_aware_slt.yaml
│   └── jobs/                   # Runnable job configs
│       ├── youtube_asl/
│       │   ├── mediapipe.yaml
│       │   ├── mmpose.yaml
│       │   └── video.yaml
│       └── how2sign/
│           ├── mediapipe.yaml
│           ├── mmpose.yaml
│           └── video.yaml
├── src/signdata/
│   ├── __main__.py             # CLI entry point
│   ├── cli.py                  # Argument parsing
│   ├── registry.py             # Component registry
│   ├── config/                 # YAML loading and Pydantic schema
│   ├── pipeline/               # Recipes, runner, and context
│   ├── datasets/               # Dataset adapters
│   ├── detection/              # Detection backends
│   ├── pose/                   # Pose extractors and variants
│   ├── processors/             # Pipeline processors
│   ├── presets/                # Keypoint presets
│   └── utils/                  # Video, file, and text utilities
├── resources/                  # Shipped model config assets
├── docs/                       # Documentation
├── assets/                     # Video ID lists and small assets
├── tests/
│   ├── integration/
│   └── unit/
└── requirements.txt
```

## Adding a New Dataset

1. Create a class in `src/signdata/datasets/` decorated with `@register_dataset`:

```python
from signdata.datasets.base import BaseDataset
from signdata.registry import register_dataset


@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    name = "my_dataset"

    @classmethod
    def validate_config(cls, config):
        pass
```

2. Import it in `src/signdata/datasets/__init__.py` so the decorator runs at startup.

3. Add a job YAML under `configs/jobs/my_dataset/`, for example `configs/jobs/my_dataset/mediapipe.yaml`. See [configuration reference](docs/configuration.md#minimal-working-config).

4. Add dataset documentation to `docs/datasets.md`.

## Adding a New Processor

1. Create a class in the appropriate `src/signdata/processors/<group>/` package decorated with `@register_processor`:

```python
from signdata.processors.base import BaseProcessor
from signdata.registry import register_processor


@register_processor("my_step")
class MyProcessor(BaseProcessor):
    name = "my_step"

    def run(self, context):
        return context

    def validate(self, context) -> bool:
        return True
```

2. Import it in the matching package `__init__.py`, such as `src/signdata/processors/pose/__init__.py` or `src/signdata/processors/video/__init__.py`. Re-export it from `src/signdata/processors/__init__.py` if the top-level package should expose it.

3. Add the stage to the relevant recipe in `src/signdata/pipeline/recipes.py`, or gate it through `config.stage_config` if it should remain optional.

## Adding a New Extractor

1. Create a class in `src/signdata/pose/` or one of its backend packages decorated with `@register_extractor`:

```python
from signdata.pose.base import LandmarkExtractor
from signdata.registry import register_extractor


@register_extractor("my_extractor")
class MyExtractor(LandmarkExtractor):
    def process_frame(self, frame):
        ...

    def close(self):
        ...
```

2. Import it in `src/signdata/pose/__init__.py`.

3. Set `extractor.name: my_extractor` in your job YAML.

## Running Tests

```bash
python -m pytest -q tests
```

See [architecture docs](docs/architecture.md) for how the registry, recipes, and pipeline runner fit together.
