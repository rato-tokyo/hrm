"""
LEGO: Layered Ensemble with Gradual Optimization

Two-phase training with hard example mining and CALM-style early exit inference.
"""

from .model import LEGOLLM
from .block import LEGOBlock
from .exit_classifier import ExitClassifier
from .modules import TransformerBlock
from .trainer import train_block
from .model_trainer import train_legollm, evaluate_legollm, create_sequence_data
from .config import TrainerConfig, ExperimentConfig
from .data import SequenceData, create_wikitext_dataloaders
from .utils import set_seed, get_device

__version__ = "0.11.0"

__all__ = [
    # Core
    'LEGOLLM',
    'LEGOBlock',
    'ExitClassifier',
    'TransformerBlock',
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
