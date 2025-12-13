"""
LEGO: Layered Ensemble with Gradual Optimization

Two-phase training with hard example mining and early exit inference.
"""

from .block import LEGOBlock
from .transformer import LEGOTransformer
from .config import ExperimentConfig
from .data import TrainingData, create_wikitext_dataloaders
from .utils import (
    set_seed,
    get_device,
    create_synthetic_data,
)

__version__ = "0.3.0"

__all__ = [
    # Core
    'LEGOBlock',
    'LEGOTransformer',
    # Config
    'ExperimentConfig',
    # Data
    'TrainingData',
    'create_wikitext_dataloaders',
    # Utilities
    'set_seed',
    'get_device',
    'create_synthetic_data',
]
