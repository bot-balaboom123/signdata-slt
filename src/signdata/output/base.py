"""Base output ABC."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..config.schema import Config

if TYPE_CHECKING:
    from ..pipeline.context import PipelineContext


class BaseOutput(ABC):
    """Abstract base class for output formatters."""

    name: str

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"signdata.output.{self.name}")

    @abstractmethod
    def run(self, context: "PipelineContext") -> "PipelineContext":
        """Execute this output step. Return updated context."""
        ...
