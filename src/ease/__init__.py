"""
EASE: Efficient Adaptive Supervision for Early Exit

Staged Deep Supervision (SDS)ベースの統一訓練フレームワーク。

Core Concept:
- Stage: 訓練の1つのフェーズ（どの層を、どのデータで、どう訓練するか）
- すべての訓練戦略をStageの組み合わせとして表現

Training Strategies:
1. Standard: 最終層のみ（1 stage）
2. Deep Supervision: 全層均等（1 stage, all layers）
3. ASHEM: Hard example mining（2 stages, progressive layers, filtered data）

Usage:
    from ease import (
        DeepSupervisionTransformer,
        StagedTrainer,
        StageConfig,
        StagedDSConfig,
        create_ashem_config,
    )

    # ASHEMの例
    config = create_ashem_config(
        phase1_layers=2,
        phase2_layers=4,
        vocab_size=69830
    )

    trainer = StagedTrainer(config, device='cuda')
    model = DeepSupervisionTransformer(vocab_size=69830, dim=64, num_layers=4)

    # Stage 1: 浅層モデル訓練
    trainer.train_stage(config.stages[0], model, train_data, val_data)

    # Stage 2: Hard examples収集 + 深層モデル訓練
    threshold = compute_confidence_threshold(model, val_data, 0.5, 'cuda', 2)
    hard_examples = collect_hard_examples(model, val_data, threshold, 'cuda', 2)
    hard_batches = create_hard_example_loader(hard_examples, 64)
    trainer.train_stage(config.stages[1], model, hard_batches, val_data)

References:
- Staged Deep Supervision: 本フレームワーク
- Deep Supervision: Lee et al., 2015
- Early Exit: Teerapittayanon et al., 2016
- ASHEM: Adaptive Supervision via Hard Example Mining（本研究）
"""

from .models import (
    StandardTransformer,
    DeepSupervisionTransformer,
)
from .staged_ds import (
    StageConfig,
    StagedDSConfig,
    StagedTrainer,
    compute_confidence,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    create_standard_config,
    create_deep_supervision_config,
    create_ashem_config,
)
from .modules import (
    RMSNorm,
    RotaryPositionalEmbedding,
    MultiHeadAttention,
    GatedLinearUnit,
    TransformerBlock,
)

__version__ = "0.3.0"  # Stageベース完全移行

__all__ = [
    # Models
    'StandardTransformer',
    'DeepSupervisionTransformer',
    # Staged DS
    'StageConfig',
    'StagedDSConfig',
    'StagedTrainer',
    'compute_confidence',
    'compute_confidence_threshold',
    'collect_hard_examples',
    'create_hard_example_loader',
    'create_standard_config',
    'create_deep_supervision_config',
    'create_ashem_config',
    # Modules
    'RMSNorm',
    'RotaryPositionalEmbedding',
    'MultiHeadAttention',
    'GatedLinearUnit',
    'TransformerBlock',
]
