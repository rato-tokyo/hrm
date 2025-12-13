"""
LEGO: Layered Ensemble with Gradual Optimization

レゴブロックのようにPhase（層グループ）を組み合わせる柔軟な訓練アーキテクチャ。

Core Concepts:
- PhaseConfig: 1つのフェーズ（層ブロック）の訓練設定
- LEGOConfig: 複数フェーズの訓練設定（Cascading方式）
- LEGOTrainer: Cascading LEGO訓練を実行

Training Strategies:
1. Standard: Final layer only (create_standard_config + Trainer)
2. LEGO: Multi-phase cascading training (LEGOConfig + LEGOTrainer)

Usage:
    from ease import (
        StandardTransformer,
        DeepSupervisionTransformer,
        LEGOConfig,
        PhaseConfig,
        LEGOTrainer,
    )

    # Define multi-phase configuration
    config = LEGOConfig(
        phases=[
            PhaseConfig(layers=(1, 2), lr=1e-3, patience=1),
            PhaseConfig(layers=(3, 4), lr=1e-4, patience=3),
        ],
        hard_example_ratio=0.5,
    )

    # Train with cascading phases
    trainer = LEGOTrainer(config, vocab_size=10000, device='cuda')
    result = trainer.train(model, train_loader, val_loader)
"""

# Types (shared across modules)
from .types import (
    DataBatch,
    HardBatch,
    HardExamples,
    EvalStats,
    TrainingHistory,
    PhaseHistory,
    LEGOResult,
)

# Models
from .models import (
    StandardTransformer,
    DeepSupervisionTransformer,
)

# Standard Trainer (for simple training)
from .trainer import (
    StageConfig,
    TrainingConfig,
    Trainer,
    create_standard_config,
)

# LEGO (multi-phase cascading training)
from .lego import (
    PhaseConfig,
    LEGOConfig,
    LEGOTrainer,
    # Utility functions (for advanced usage)
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    train_upper_layers,
    evaluate_on_hard_examples,
)

__version__ = "0.4.0"

__all__ = [
    # Types
    'DataBatch',
    'HardBatch',
    'HardExamples',
    'EvalStats',
    'TrainingHistory',
    'PhaseHistory',
    'LEGOResult',
    # Models
    'StandardTransformer',
    'DeepSupervisionTransformer',
    # Standard Trainer
    'StageConfig',
    'TrainingConfig',
    'Trainer',
    'create_standard_config',
    # LEGO
    'PhaseConfig',
    'LEGOConfig',
    'LEGOTrainer',
    'compute_confidence_threshold',
    'collect_hard_examples',
    'create_hard_example_loader',
    'train_upper_layers',
    'evaluate_on_hard_examples',
]
