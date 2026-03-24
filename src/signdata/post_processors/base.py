"""Base post-processor ABC."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..config.schema import Config

if TYPE_CHECKING:
    from ..pipeline.context import PipelineContext


class BasePostProcessor(ABC):
    """Abstract base class for post-processing recipes."""

    name: str

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"signdata.post.{self.name}")

    @abstractmethod
    def run(self, context: "PipelineContext") -> "PipelineContext":
        """Execute this post-processing step. Return updated context."""
        ...
