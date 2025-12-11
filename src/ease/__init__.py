"""
EASE: Efficient Asymmetric Supervision for Early-Exit Transformers

A unified training framework for Early-Exit Transformers that supports:
- Deep Supervision (Lee et al., 2015)
- Auxiliary Loss Training (Elbayad et al., 2020)
- Asymmetric Auxiliary Loss (Ours)
- Discriminative Fine-Tuning (Howard & Ruder, 2018)
- Learning Rate Curriculum (Croitoru et al., 2024)

Usage:
    from ease import ConfidenceRoutedTransformer, UniversalTrainer, UniversalConfig, PRESETS

    # Create model
    model = ConfidenceRoutedTransformer(vocab_size=1000, dim=64, num_layers=3)

    # Use preset configuration
    config = PRESETS['asymmetric']  # α=0.7, L2=0, routing_threshold=0.95

    # Or create custom configuration
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

from .models import StandardTransformer, ConfidenceRoutedTransformer
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
    'ConfidenceRoutedTransformer',
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
