"""
EASE: Efficient Asymmetric Supervision for Early-Exit Transformers

A unified training framework for Early-Exit Transformers that supports:
- DEED (Tang et al., 2023) - Deep Supervision + Dynamic Early Exit
- Auxiliary Loss Training (Elbayad et al., 2020)
- Discriminative Fine-Tuning (Howard & Ruder, 2018)

Models:
- DEEDTransformer: α-weighted loss distribution (all tokens → both losses)
- TokenRoutedTransformer: Mask-based routing (each token → one loss, no α needed)
- MoDTransformer: Top-k token selection for dynamic compute
- StandardTransformer: Baseline model

Usage:
    from ease import DEEDTransformer, UniversalTrainer, UniversalConfig

    # Create model
    model = DEEDTransformer(vocab_size=1000, dim=64, num_layers=3)

    # Configure α-weighted loss distribution
    config = UniversalConfig(
        layer_weights={1: 0.7, 2: 0, 3: 0.3},
        routing_threshold=0.95,
    )

    # Create trainer
    trainer = UniversalTrainer(config, vocab_size=1000)
    optimizer = trainer.create_optimizer(model, base_lr=1e-3)

    # Train
    loss, weights = trainer.train_epoch(model, train_batches, optimizer)

    # Evaluate
    stats = trainer.evaluate(model, val_batches)
"""

from .models import (
    StandardTransformer,
    DEEDTransformer,
    MoDTransformer,
    TokenRoutedTransformer,
)
from .trainer import UniversalConfig, UniversalTrainer, AlphaSchedule, PRESETS
from .modules import (
    RMSNorm,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    GatedLinearUnit,
    TransformerBlock,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    'StandardTransformer',
    'DEEDTransformer',
    'MoDTransformer',
    'TokenRoutedTransformer',
    # Trainer
    'UniversalConfig',
    'UniversalTrainer',
    'AlphaSchedule',
    'PRESETS',
    # Modules
    'RMSNorm',
    'RotaryPositionalEmbedding',
    'MultiHeadAttention',
    'GatedLinearUnit',
    'TransformerBlock',
]
