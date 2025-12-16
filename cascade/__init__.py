"""
CASCADE: Confidence-Aware Sequential Compute Allocation for Dynamic Exit

複数のLLMを統合し、Early Exitで効率的にルーティングするフレームワーク。
- 任意のHugging Face LLMをLLMクラスでラップ
- Early Exit機能により、簡単なトークンは前段LLMで処理完了
- Hard tokens（難しいトークン）だけを後段に渡す

Hugging Face Transformersとの統合:
- AutoTokenizer, AutoModelForCausalLMを使用
- TrainingArgumentsを直接使用
- datasets.Datasetとの相互変換
"""

from .ensemble import Ensemble
from .llm import LLM
from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim
from .llm_trainer import train_llm
from .llm_trainer_simple import train_llm_simple
from .llm_evaluator import compute_ppl, evaluate_llm
from .ensemble_trainer import train_ensemble, create_sequence_data
from .config import CascadeConfig, ExperimentConfig
from .sequence_data import SequenceData
from .dataloader import (
    create_wikitext_dataloaders,
    create_dataset_from_tokenizer,
)
from .utils import set_seed, get_device

__version__ = "0.21.0"

__all__ = [
    # コア
    'Ensemble',
    'LLM',
    # Exit関数
    'ExitFn',
    'default_exit_fn',
    'compute_cos_sim',
    # 訓練
    'train_ensemble',
    'train_llm',
    'train_llm_simple',
    'create_sequence_data',
    # 評価
    'compute_ppl',
    'evaluate_llm',
    # 設定
    'CascadeConfig',
    'ExperimentConfig',
    # データ
    'SequenceData',
    'create_wikitext_dataloaders',
    'create_dataset_from_tokenizer',
    # ユーティリティ
    'set_seed',
    'get_device',
]
