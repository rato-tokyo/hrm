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
# Dual-Context Attention（L0/L1 2層コンテキスト）
from .dual_context_attention import (
    DCAOutput,
    DualContextState,
    DualContextMemory,
    DualContextAttention,
    DualContextLM,
)
# DCA-LLM（推論対応）
from .dca_llm import (
    DCALLMOutput,
    DCALLM,
    create_dca_llm,
    create_dca_llm_from_scratch,
    # Integrated DCA (DCAを内部に統合)
    IntegratedDCABlock,
    IntegratedDCALLM,
    create_integrated_dca_llm,
)

__version__ = "0.33.0"

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
    # Dual-Context Attention
    'DCAOutput',
    'DualContextState',
    'DualContextMemory',
    'DualContextAttention',
    'DualContextLM',
    # DCA-LLM
    'DCALLMOutput',
    'DCALLM',
    'create_dca_llm',
    'create_dca_llm_from_scratch',
    # Integrated DCA
    'IntegratedDCABlock',
    'IntegratedDCALLM',
    'create_integrated_dca_llm',
]
