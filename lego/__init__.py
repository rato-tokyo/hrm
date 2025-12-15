"""
LEGO: Layered Ensemble with Gradual Optimization

Two-phase training with hard example mining and early exit inference.
Exit decision is made by exit_fn using hidden_history from all layers.
"""

from .model import LEGOLLM
from .block import LEGOBlock, default_exit_fn, ExitFn
from .modules import TransformerBlock
from .trainer import train_block
from .model_trainer import train_legollm, evaluate_legollm, create_sequence_data
from .config import TrainerConfig, ExperimentConfig
from .data import SequenceData, create_wikitext_dataloaders
from .utils import set_seed, get_device

__version__ = "0.12.0"

__all__ = [
    # Core
    'LEGOLLM',
    'LEGOBlock',
    'TransformerBlock',
    # Exit function
    'default_exit_fn',
    'ExitFn',
    # Training - LEGOLLM
    'train_legollm',
    'evaluate_legollm',
    'create_sequence_data',
    # Training - LEGOBlock
    'train_block',
    # Config
    'TrainerConfig',
    'ExperimentConfig',
    # Data
    'SequenceData',
    'create_wikitext_dataloaders',
    # Utilities
    'set_seed',
    'get_device',
]
