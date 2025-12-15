"""
CASCADE: Confidence-Aware Sequential Compute Allocation for Dynamic Exit

複数のLLMを統合し、Early Exitで効率的にルーティングするフレームワーク。
- 任意のLLMをLLMクラスでラップ
- Early Exit機能により、簡単なトークンは前段LLMで処理完了
- Hard tokens（難しいトークン）だけを後段に渡す
"""

from .ensemble import Ensemble
from .llm import LLM
from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim
from .modules import TransformerBlock
from .llm_trainer import train_llm
from .ensemble_trainer import train_ensemble, create_sequence_data
from .evaluator import evaluate_ensemble
from .config import TrainerConfig, ExperimentConfig
from .sequence_data import SequenceData
from .dataloader import create_wikitext_dataloaders
from .utils import set_seed, get_device

__version__ = "0.17.0"

__all__ = [
    # コア
    'Ensemble',
    'LLM',
    'TransformerBlock',
    # Exit関数
    'ExitFn',
    'default_exit_fn',
    'compute_cos_sim',
    # 訓練
    'train_ensemble',
    'train_llm',
    'create_sequence_data',
    # 評価
    'evaluate_ensemble',
    # 設定
    'TrainerConfig',
    'ExperimentConfig',
    # データ
    'SequenceData',
    'create_wikitext_dataloaders',
    # ユーティリティ
    'set_seed',
    'get_device',
]
