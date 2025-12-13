"""
LEGO: Layered Ensemble with Gradual Optimization

A modular training framework with 2 core options.

Base Models:
- StandardTransformer: Final layer loss only
- DeepSupervisionTransformer: Loss at all layers with early exit support

Core Options (via TrainingConfig):
- stages: Stage-based training configuration
- routing_threshold: Early exit at inference

Training Strategies:
1. Standard: Final layer only (1 stage)
2. Deep Supervision: All layers equally (all stages)
3. Hard Example Mining: 2-stage training with hard example focus

Usage:
    from lego import DeepSupervisionTransformer, Trainer, TrainingConfig, StageConfig

    # Create model
    model = DeepSupervisionTransformer(vocab_size=1000, dim=64, num_layers=3)

    # Configure training (2 core options)
    config = TrainingConfig(
        stages=[
            StageConfig(layers=(1, 2), loss_weight=0.7),
            StageConfig(layers=(3, 3), loss_weight=0.3),
        ],
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
- Deep Supervision: Lee et al., 2015
- Early Exit: Teerapittayanon et al., 2016
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
"""

from .models import (
    StandardTransformer,
    DeepSupervisionTransformer,
)
from .trainer import (
    StageConfig,
    TrainingConfig,
    Trainer,
    create_standard_config,
    create_deep_supervision_config,
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
    'StandardTransformer',
    'DeepSupervisionTransformer',
    # Trainer
    'StageConfig',
    'TrainingConfig',
    'Trainer',
    'create_standard_config',
    'create_deep_supervision_config',
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
