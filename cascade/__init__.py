"""
CASCADE: Confidence-Aware Sequential Compute Allocation for Dynamic Exit

複数のLLMを統合し、Early Exitで効率的にルーティングするフレームワーク。
- 任意のHugging Face LLMをLLMクラスでラップ
- Early Exit機能により、簡単なトークンは前段LLMで処理完了
- Hard tokens（難しいトークン）だけを後段に渡す

Hugging Face Transformersとの完全統合:
- AutoTokenizer, AutoModelForCausalLMを使用
- TrainingArgumentsを直接使用
- datasets.Datasetを直接使用
- HF Trainerで訓練
"""

from .ensemble import Ensemble
from .llm import LLM
from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim
from .cascade_trainer import CascadeTrainer, create_initial_dataset
from .ensemble_trainer import train_ensemble
from .llm_evaluator import compute_ppl, evaluate_llm
from .cascade_dataset import (
    create_cascade_dataset,
    get_dataset_info,
    iterate_batches,
    collect_hard_tokens_from_dataset,
    transform_dataset,
    reconstruct_sequences,
    create_empty_dataset,
)
from .config import CascadeConfig, ExperimentConfig
from .dataloader import (
    create_wikitext_dataloaders,
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
    PRETRAINED_MODELS,
)

__version__ = "0.23.0"

__all__ = [
    # コア
    'Ensemble',
    'LLM',
    # Exit関数
    'ExitFn',
    'default_exit_fn',
    'compute_cos_sim',
    # 訓練（HF Trainerベース）
    'CascadeTrainer',
    'train_ensemble',
    'create_initial_dataset',
    # Dataset操作（HF Dataset直接使用）
    'create_cascade_dataset',
    'get_dataset_info',
    'iterate_batches',
    'collect_hard_tokens_from_dataset',
    'transform_dataset',
    'reconstruct_sequences',
    'create_empty_dataset',
    # 評価
    'compute_ppl',
    'evaluate_llm',
    # 設定
    'CascadeConfig',
    'ExperimentConfig',
    # データローダー
    'create_wikitext_dataloaders',
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
    'PRETRAINED_MODELS',
]
