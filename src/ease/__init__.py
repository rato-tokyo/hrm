"""
LEGO: Layered Ensemble with Gradual Optimization

レゴブロックのようにStage（層グループ）を組み合わせる柔軟な訓練アーキテクチャ。

Base Models:
- StandardTransformer: Final layer loss only
- DeepSupervisionTransformer: Loss at all layers with early exit support

Core Options (via TrainingConfig):
- stages: Stage-based training configuration (LEGO blocks)
- routing_threshold: Early exit at inference

Training Strategies:
1. Standard LEGO: Final layer only (1 stage block)
2. Deep Supervision LEGO: All layers equally (all stage blocks)
3. ASHEM LEGO: Hard example mining with 2-stage blocks

Usage:
    from ease import DeepSupervisionTransformer, Trainer, TrainingConfig, StageConfig

    # Create model
    model = DeepSupervisionTransformer(vocab_size=1000, dim=64, num_layers=3)

    # Configure training (LEGO's 2 core options)
    config = TrainingConfig(
        stages=[
            StageConfig(layers=(1, 2), loss_weight=0.7),  # Stage Block 1
            StageConfig(layers=(3, 3), loss_weight=0.3),  # Stage Block 2
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
- ASHEM: Adaptive Supervision via Hard Example Mining
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
from .ashem import (
    ASHEMConfig,
    compute_confidence,
    compute_confidence_threshold,
    collect_hard_examples,
    freeze_lower_layers,
    get_trainable_params_info,
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
    # ASHEM
    'ASHEMConfig',
    'compute_confidence',
    'compute_confidence_threshold',
    'collect_hard_examples',
    'freeze_lower_layers',
    'get_trainable_params_info',
    # Modules
    'RMSNorm',
    'RotaryPositionalEmbedding',
    'MultiHeadAttention',
    'GatedLinearUnit',
    'TransformerBlock',
]
