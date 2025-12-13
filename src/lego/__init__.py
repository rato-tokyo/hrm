"""
LEGO: Layered Ensemble with Gradual Optimization

A modular training framework with 2 core options.

Model:
- LEGOTransformer: Unified model supporting standard and early exit

Core Options (via TrainingConfig):
- output_layer: Which layer to compute loss
- routing_threshold: Early exit at inference

Training Strategies:
1. Standard: Final layer only
2. Hard Example Mining: 2-phase training with hard example focus

Usage:
    from lego import LEGOTransformer, Trainer, TrainingConfig

    # Create model
    model = LEGOTransformer(vocab_size=1000, dim=64, num_layers=3)

    # Configure training
    config = TrainingConfig(
        output_layer=3,
        routing_threshold=0.95,
    )

    # Create trainer
    trainer = Trainer(config, vocab_size=1000)
    optimizer = trainer.create_optimizer(model, base_lr=1e-3)

    # Train
    loss = trainer.train_epoch(model, train_batches, optimizer)

    # Evaluate
    stats = trainer.evaluate(model, val_batches)

References:
- LEGO: Layered Ensemble with Gradual Optimization
- Early Exit: Teerapittayanon et al., 2016
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
"""

from .models import LEGOTransformer
from .trainer import (
    TrainingConfig,
    Trainer,
    create_standard_config,
)
from .utils import (
    LEGOConfig,
    compute_confidence,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    train_upper_layers,
    evaluate_on_hard_examples,
)
from .modules import (
    RMSNorm,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    GatedLinearUnit,
    TransformerBlock,
)

__version__ = "0.2.0"

__all__ = [
    # Models
    'LEGOTransformer',
    # Trainer
    'TrainingConfig',
    'Trainer',
    'create_standard_config',
    # LEGO Config
    'LEGOConfig',
    'compute_confidence',
    'compute_confidence_threshold',
    'collect_hard_examples',
    'create_hard_example_loader',
    'train_upper_layers',
    'evaluate_on_hard_examples',
    # Modules
    'RMSNorm',
    'RotaryPositionalEmbedding',
    'MultiHeadAttention',
    'GatedLinearUnit',
    'TransformerBlock',
]
