"""
LEGO: Layered Ensemble with Gradual Optimization

Two-phase training with hard example mining and early exit inference.
"""

from .model import LEGOLLM
from .block import LEGOBlock
from .modules import TransformerBlock, TransformerLayer
from .trainer import train_block
from .config import TrainerConfig, ExperimentConfig
from .data import TrainingData, create_wikitext_dataloaders
from .utils import (
    set_seed,
    get_device,
    create_synthetic_data,
)

__version__ = "0.6.0"

__all__ = [
    # Core - LEGO components
    'LEGOLLM',
    'LEGOBlock',
    # Core - Standard transformer components
    'TransformerBlock',
    'TransformerLayer',
    # Training
    'train_block',
    'TrainerConfig',
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
