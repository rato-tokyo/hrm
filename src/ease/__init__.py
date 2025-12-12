"""
LEGO: Layered Ensemble with Gradual Optimization

レゴブロックのようにStage（層グループ）を組み合わせる柔軟な訓練アーキテクチャ。

Core Options (via TrainingConfig):
1. stages: Stage-based training configuration (LEGO blocks)
2. routing_threshold: Early exit at inference

Training Strategies:
1. Standard LEGO: Final layer only (create_standard_config)
2. ASHEM LEGO: Hard example mining with 2-stage blocks

Usage:
    from ease import (
        StandardTransformer,
        DeepSupervisionTransformer,
        Trainer,
        TrainingConfig,
        StageConfig,
        create_standard_config,
    )

    # Phase 1: Train shallow model
    model = StandardTransformer(vocab_size=1000, dim=64, num_layers=2)
    config = create_standard_config(num_layers=2)
    trainer = Trainer(config, vocab_size=1000)

    # Phase 2: Extend and train on hard examples (see ASHEM functions)

References:
- LEGO: Layered Ensemble with Gradual Optimization
- ASHEM: Adaptive Supervision via Hard Example Mining
"""

# Types (shared across modules)
from .types import (
    DataBatch,
    HardBatch,
    HardExamples,
    EvalStats,
    TrainingHistory,
)

# Models
from .models import (
    StandardTransformer,
    DeepSupervisionTransformer,
)

# Trainer
from .trainer import (
    StageConfig,
    TrainingConfig,
    Trainer,
    create_standard_config,
)

# ASHEM
from .ashem import (
    ASHEMConfig,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    train_upper_layers,
    evaluate_on_hard_examples,
)

__version__ = "0.3.0"

__all__ = [
    # Types
    'DataBatch',
    'HardBatch',
    'HardExamples',
    'EvalStats',
    'TrainingHistory',
    # Models
    'StandardTransformer',
    'DeepSupervisionTransformer',
    # Trainer
    'StageConfig',
    'TrainingConfig',
    'Trainer',
    'create_standard_config',
    # ASHEM
    'ASHEMConfig',
    'compute_confidence_threshold',
    'collect_hard_examples',
    'create_hard_example_loader',
    'train_upper_layers',
    'evaluate_on_hard_examples',
]
