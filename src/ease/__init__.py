"""
LEGO: Layered Ensemble with Gradual Optimization

レゴブロックのようにPhase（層グループ）を組み合わせる柔軟な訓練アーキテクチャ。

Core Concepts:
- LEGOTransformer: 統一されたTransformerモデル
- PhaseConfig: 1つのフェーズ（層ブロック）の訓練設定
- LEGOConfig: 複数フェーズの訓練設定（Cascading方式）
- LEGOTrainer: Cascading LEGO訓練を実行

Usage:
    from ease import LEGOTransformer, LEGOConfig, PhaseConfig, LEGOTrainer

    # Create model
    model = LEGOTransformer(vocab_size=10000, num_layers=4, exit_layer=2)

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

# Model
from .models import (
    LEGOTransformer,
    # Backward compatibility aliases
    StandardTransformer,
    DeepSupervisionTransformer,
)

# Standard Trainer (for evaluation and simple training)
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

__version__ = "0.5.0"

__all__ = [
    # Types
    'DataBatch',
    'HardBatch',
    'HardExamples',
    'EvalStats',
    'TrainingHistory',
    'PhaseHistory',
    'LEGOResult',
    # Model
    'LEGOTransformer',
    'StandardTransformer',  # Backward compatibility
    'DeepSupervisionTransformer',  # Backward compatibility
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
