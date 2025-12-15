"""
LEGO: Layered Ensemble with Gradual Optimization

既存のLLMに新しいLLMを段階的に追加・統合するフレームワーク。
- LLM 0は既存の訓練済みLLM
- LLM 1以降は未学習で、hard tokensのみで訓練
- Early Exit機能により、簡単なトークンは前段LLMで処理完了
"""

from .lego_ensemble import LEGOEnsemble
from .early_exit_llm import EarlyExitLLM
from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim
from .modules import TransformerBlock
from .llm_trainer import train_llm
from .ensemble_trainer import train_ensemble, create_sequence_data
from .evaluator import evaluate_ensemble
from .config import TrainerConfig, ExperimentConfig
from .sequence_data import SequenceData
from .dataloader import create_wikitext_dataloaders
from .utils import set_seed, get_device

__version__ = "0.15.0"

__all__ = [
    # コア
    'LEGOEnsemble',
    'EarlyExitLLM',
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
