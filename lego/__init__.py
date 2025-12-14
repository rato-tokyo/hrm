"""
LEGO: Layered Ensemble with Gradual Optimization

Two-phase training with hard example mining and early exit inference.
"""

from .model import LEGOLLM
from .block import LEGOBlock
from .modules import TransformerBlock
from .trainer import train_block
from .config import TrainerConfig, ExperimentConfig
from .data import SequenceData, create_wikitext_dataloaders
from .utils import set_seed, get_device

__version__ = "0.8.0"

__all__ = [
    # Core
    'LEGOLLM',
    'LEGOBlock',
    'TransformerBlock',
    # Training
    'train_block',
    'TrainerConfig',
    'ExperimentConfig',
    # Data
    'SequenceData',
    'create_wikitext_dataloaders',
    # Utilities
    'set_seed',
    'get_device',
]
