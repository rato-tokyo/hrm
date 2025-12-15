"""
LEGO: Layered Ensemble with Gradual Optimization

Two-phase training with hard example mining and early exit inference.
Exit decision is made by exit_fn using hidden_history from all layers.
"""

from .model import LEGOLLM
from .block import LEGOBlock
from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim
from .modules import TransformerBlock
from .trainer import train_block
from .model_trainer import train_legollm, create_sequence_data
from .evaluator import evaluate_legollm
from .config import TrainerConfig, ExperimentConfig
from .data import SequenceData
from .dataloader import create_wikitext_dataloaders
from .utils import set_seed, get_device

__version__ = "0.13.0"

__all__ = [
    # Core
    'LEGOLLM',
    'LEGOBlock',
    'TransformerBlock',
    # Exit function
    'ExitFn',
    'default_exit_fn',
    'compute_cos_sim',
    # Training
    'train_legollm',
    'train_block',
    'create_sequence_data',
    # Evaluation
    'evaluate_legollm',
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
