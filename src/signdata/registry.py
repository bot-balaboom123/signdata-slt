"""Component registry for datasets, processors, post-processors, and output."""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .datasets.base import BaseDataset
    from .processors.base import BaseProcessor
    from .post_processors.base import BasePostProcessor
    from .output.base import BaseOutput

DATASET_REGISTRY: Dict[str, Type["BaseDataset"]] = {}
PROCESSOR_REGISTRY: Dict[str, Type["BaseProcessor"]] = {}
POST_PROCESSOR_REGISTRY: Dict[str, Type["BasePostProcessor"]] = {}
OUTPUT_REGISTRY: Dict[str, Type["BaseOutput"]] = {}


def register_dataset(name: str):
    """Register a dataset class under the given name."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def register_processor(name: str):
    """Register a processor class under the given name."""
    def decorator(cls):
        PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_post_processor(name: str):
    """Register a post-processor class under the given name."""
    def decorator(cls):
        POST_PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_output(name: str):
    """Register an output class under the given name."""
    def decorator(cls):
        OUTPUT_REGISTRY[name] = cls
        return cls
    return decorator
