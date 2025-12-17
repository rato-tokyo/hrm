"""
CASCADE: Confidence-Aware Sequential Compute Allocation for Dynamic Exit

DCA-LLM v2: Dual-Context Attention を内蔵したTransformer言語モデル。

主要コンポーネント:
- IntegratedDCABlock: L0/L1 2層コンテキストAttentionを内蔵したTransformerブロック
- IntegratedDCALLM: DCAを統合した言語モデル
- create_integrated_dca_llm: ファクトリ関数
"""

from .dataloader import (
    create_wikitext_dataloaders,
    create_alpaca_dataloaders,
    create_dataset_from_tokenizer,
)
from .utils import set_seed, get_device
from .model_registry import (
    ModelSpec,
    ModelRegistry,
    get_registry,
    load_pretrained,
    list_available_models,
    create_small_llm,
    create_llm_from_base,
    PRETRAINED_MODELS,
)
# DCA-LLM v2（L0/L1 2層コンテキスト）
from .dca_llm import (
    DCALLMOutput,
    IntegratedDCABlock,
    IntegratedDCALLM,
    create_integrated_dca_llm,
)
# 訓練ユーティリティ
from .trainer_utils import (
    compute_ppl,
    train_epoch,
    train_model,
    get_memory_usage,
    create_baseline_gpt2,
)

__version__ = "0.34.0"

__all__ = [
    # データローダー
    'create_wikitext_dataloaders',
    'create_alpaca_dataloaders',
    'create_dataset_from_tokenizer',
    # ユーティリティ
    'set_seed',
    'get_device',
    # モデルレジストリ
    'ModelSpec',
    'ModelRegistry',
    'get_registry',
    'load_pretrained',
    'list_available_models',
    'create_small_llm',
    'create_llm_from_base',
    'PRETRAINED_MODELS',
    # DCA-LLM v2
    'DCALLMOutput',
    'IntegratedDCABlock',
    'IntegratedDCALLM',
    'create_integrated_dca_llm',
    # 訓練ユーティリティ
    'compute_ppl',
    'train_epoch',
    'train_model',
    'get_memory_usage',
    'create_baseline_gpt2',
]
