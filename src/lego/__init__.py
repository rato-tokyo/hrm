"""
LEGO: Layered Ensemble with Gradual Optimization

Two-phase training with hard example mining and early exit inference.
"""

from .models import LEGOBlock, LEGOTransformer
from .trainer import Trainer
from .config import ExperimentConfig
from .data import create_wikitext_dataloaders
from .utils import (
    set_seed,
    get_device,
    create_synthetic_data,
    compute_routing_cost,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    train_new_block,
    evaluate_on_hard_examples,
)
from .modules import (
    RMSNorm,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    GatedLinearUnit,
    TransformerBlock,
)

__version__ = "0.2.1"

__all__ = [
    # Models
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
    'compute_routing_cost',
    'compute_confidence_threshold',
    'collect_hard_examples',
    'create_hard_example_loader',
    'train_new_block',
    'evaluate_on_hard_examples',
    # Modules
    'RMSNorm',
    'RotaryPositionalEmbedding',
    'MultiHeadAttention',
    'GatedLinearUnit',
    'TransformerBlock',
]
