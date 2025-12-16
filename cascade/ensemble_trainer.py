"""
CASCADEフレームワーク - Ensemble訓練

CascadeTrainerを使用したEnsemble訓練関数。
"""

from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING

from datasets import Dataset
from transformers import TrainingArguments

if TYPE_CHECKING:
    from .ensemble import Ensemble

from .config import CascadeConfig
from .cascade_trainer import CascadeTrainer


def train_ensemble(
    ensemble: "Ensemble",
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: TrainingArguments,
    cascade_config: CascadeConfig,
) -> Dict[str, Any]:
    """
    Ensembleの全LLMを順番に訓練。

    CascadeTrainerを使用してHF Trainerベースの訓練を実行。

    訓練フロー:
    1. LLM 0を全訓練データで訓練 → hard tokens収集
    2. LLM 1をhard tokensで訓練 → hard tokens収集
    3. ... 全LLMまで続行

    Args:
        ensemble: 訓練するEnsemble
        train_dataset: 訓練用HF Dataset (hidden_states, labels)
        val_dataset: 検証用HF Dataset
        training_args: Hugging Face TrainingArguments
        cascade_config: CASCADE固有の設定（patience, hard_ratio, lr_decay）

    Returns:
        LLMごとの統計と全体の訓練情報を含むDict
    """
    cascade_trainer = CascadeTrainer(training_args, cascade_config)
    return cascade_trainer.train(ensemble, train_dataset, val_dataset)
