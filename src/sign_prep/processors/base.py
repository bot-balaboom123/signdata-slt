"""Base processor class for pipeline steps."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline.context import PipelineContext

from ..config.schema import Config


class BaseProcessor(ABC):
    """Abstract base class for pipeline processing steps."""

    name: str  # Must match registry key

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"sign_prep.{self.name}")

    @abstractmethod
    def run(self, context: "PipelineContext") -> "PipelineContext":
        """Execute this processing step. Return updated context."""
        pass

    def validate(self, context: "PipelineContext") -> bool:
        """Check prerequisites. Override for custom validation."""
        return True
