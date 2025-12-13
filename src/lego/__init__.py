"""
LEGO: Layered Ensemble with Gradual Optimization

Two-phase training with hard example mining and early exit inference.
"""

from .block import LEGOBlock
from .transformer import LEGOTransformer
from .trainer import Trainer
from .config import ExperimentConfig
from .data import create_wikitext_dataloaders
from .utils import (
    set_seed,
    get_device,
    create_synthetic_data,
    compute_confidence_threshold,
    collect_hard_examples,
    split_hard_examples,
    create_hard_example_loader,
)
from .modules import (
    RMSNorm,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    GatedLinearUnit,
    TransformerBlock,
)

__version__ = "0.3.0"

__all__ = [
    # Core
    'LEGOBlock',
    'LEGOTransformer',
    # Trainer
    'Trainer',
    # Config
    'ExperimentConfig',
    # Data
    'create_wikitext_dataloaders',
    # Utilities
    'set_seed',
    'get_device',
    'create_synthetic_data',
    'compute_confidence_threshold',
    'collect_hard_examples',
    'split_hard_examples',
    'create_hard_example_loader',
    # Modules
    'RMSNorm',
    'RotaryPositionalEmbedding',
    'MultiHeadAttention',
    'GatedLinearUnit',
    'TransformerBlock',
]
