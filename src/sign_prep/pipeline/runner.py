"""Pipeline runner that executes a chain of processors."""

import logging
from pathlib import Path
from typing import List

from ..config.schema import Config
from ..registry import DATASET_REGISTRY, PROCESSOR_REGISTRY
from ..processors.base import BaseProcessor
from .context import PipelineContext

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Execute a chain of processors based on configuration."""

    def __init__(self, config: Config):
        self.config = config
        self.dataset = DATASET_REGISTRY[config.dataset]()
        self.processors = self._build_processor_chain()

    def _build_processor_chain(self) -> List[BaseProcessor]:
        steps = list(self.config.pipeline.steps)

        if self.config.pipeline.start_from:
            start = self.config.pipeline.start_from
            if start not in steps:
                raise ValueError(
                    f"start_from='{start}' not found in steps: {steps}"
                )
            idx = steps.index(start)
            steps = steps[idx:]

        if self.config.pipeline.stop_at:
            stop = self.config.pipeline.stop_at
            if stop not in steps:
                raise ValueError(
                    f"stop_at='{stop}' not found in steps: {steps}"
                )
            idx = steps.index(stop)
            steps = steps[: idx + 1]

        processors = []
        for name in steps:
            if name not in PROCESSOR_REGISTRY:
                raise ValueError(
                    f"Unknown processor '{name}'. "
                    f"Available: {list(PROCESSOR_REGISTRY.keys())}"
                )
            processors.append(PROCESSOR_REGISTRY[name](self.config))
        return processors

    def run(self) -> PipelineContext:
        context = PipelineContext(
            config=self.config,
            dataset=self.dataset,
            project_root=Path(self.config.paths.root).parent.parent,
        )

        logger.info("=" * 70)
        logger.info(
            "Pipeline: dataset=%s  mode=%s  steps=%s",
            self.config.dataset,
            self.config.pipeline.mode,
            [p.name for p in self.processors],
        )
        logger.info("=" * 70)

        for processor in self.processors:
            logger.info("Running step: %s", processor.name)

            if not processor.validate(context):
                raise RuntimeError(f"Validation failed for {processor.name}")

            context = processor.run(context)
            context.completed_steps.append(processor.name)

            step_stats = context.stats.get(processor.name, {})
            logger.info("Completed: %s | Stats: %s", processor.name, step_stats)

        logger.info("Pipeline complete.")
        return context
