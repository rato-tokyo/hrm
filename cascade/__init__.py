"""
CASCADE: Confidence-Aware Sequential Compute Allocation for Dynamic Exit

複数のLLMを統合するフレームワーク。
三角形Attention方式で意味的spanを検出し、多段階で情報を圧縮。

Hugging Face Transformersとの完全統合:
- AutoTokenizer, AutoModelForCausalLMを使用
- TrainingArgumentsを直接使用
- datasets.Datasetを直接使用
- HF Trainerで訓練
"""

from .ensemble import Ensemble
from .llm import LLM, TokenTensor, HiddenTensor
from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim, compute_cos_sim_from_history
from .llm_evaluator import compute_ppl
from .cascade_dataset import (
    create_cascade_dataset,
    get_dataset_info,
    iterate_batches,
    create_empty_dataset,
)
from .config import ExperimentConfig
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
# Span検出（LTri-LLM論文ベース + 独自実装）
from .span_detector import (
    Span,
    SpanDetector,
    TriangleScoreDetector,
    RowChangeDetector,
    FixedSpanDetector,
    create_span_detector,
    spans_to_boundaries,
    aggregate_attention_maps,
)
from .span_compressor import (
    CompressedOutput,
    SpanCompressor,
    extract_span_boundaries,
    reconstruct_from_boundaries,
)
from .hierarchical_exit import (
    StageOutput,
    HierarchicalOutput,
    HierarchicalExit,
    create_hierarchical_exit,
)
# Infini-Span Memory（Local/Global分離）
from .infini_span_memory import (
    SpanInfo,
    MemoryState,
    BidirectionalSpanEncoder,
    InfiniSpanMemory,
    InfiniSpanAttention,
)

__version__ = "0.29.0"

__all__ = [
    # コア
    'Ensemble',
    'LLM',
    # 型エイリアス
    'TokenTensor',
    'HiddenTensor',
    # Exit関数
    'ExitFn',
    'default_exit_fn',
    'compute_cos_sim',
    'compute_cos_sim_from_history',
    # Dataset操作（HF Dataset直接使用）
    'create_cascade_dataset',
    'get_dataset_info',
    'iterate_batches',
    'create_empty_dataset',
    # 評価
    'compute_ppl',
    # 設定
    'ExperimentConfig',
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
    # Span検出（LTri-LLM論文ベース + 独自実装）
    'Span',
    'SpanDetector',
    'TriangleScoreDetector',
    'RowChangeDetector',
    'FixedSpanDetector',
    'create_span_detector',
    'spans_to_boundaries',
    'aggregate_attention_maps',
    # Span圧縮
    'CompressedOutput',
    'SpanCompressor',
    'extract_span_boundaries',
    'reconstruct_from_boundaries',
    # 階層的Early Exit
    'StageOutput',
    'HierarchicalOutput',
    'HierarchicalExit',
    'create_hierarchical_exit',
    # Infini-Span Memory
    'SpanInfo',
    'MemoryState',
    'BidirectionalSpanEncoder',
    'InfiniSpanMemory',
    'InfiniSpanAttention',
]
