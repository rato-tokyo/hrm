"""
LASH: Layered Adaptive Supervision Hierarchy

層を組み合わせる柔軟なフレームワーク。2つのコアオプションで全てを制御。

Base Models:
- StandardTransformer: Final layer loss only
- DeepSupervisionTransformer: Loss at all layers with early exit support

Core Options (via TrainingConfig):
- layer_weights: Layer-wise loss weights
- routing_threshold: Early exit at inference

Training Strategies:
1. Standard: Final layer only
2. Deep Supervision: All layers equally
3. ASHEM: Hard example mining with 2-stage training

Usage:
    from ease import DeepSupervisionTransformer, Trainer, TrainingConfig

    # Create model
    model = DeepSupervisionTransformer(vocab_size=1000, dim=64, num_layers=3)

    # Configure training (LASH's 2 core options)
    config = TrainingConfig(
        layer_weights={1: 0.7, 2: 0, 3: 0.3},
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
- LASH: Layered Adaptive Supervision Hierarchy
- Deep Supervision: Lee et al., 2015
- Discriminative Fine-Tuning: Howard & Ruder, 2018
- Early Exit: Teerapittayanon et al., 2016
- ASHEM: Adaptive Supervision via Hard Example Mining
"""

from .models import (
    StandardTransformer,
    DeepSupervisionTransformer,
)
from .trainer import (
    TrainingConfig,
    Trainer,
    create_standard_config,
    create_deep_supervision_config,
)
from .ashem import (
    ASHEMConfig,
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
    'TrainingConfig',
    'Trainer',
    'create_standard_config',
    'create_deep_supervision_config',
    # ASHEM
    'ASHEMConfig',
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
