"""
CASCADEフレームワーク - LLM訓練（Hugging Face Trainer使用）

LLMの訓練関数。Hugging Face TrainerとTrainingArgumentsを使用。
Hard token収集はLLM.collect_hard_tokens()で処理。

注意: このモジュールは訓練専用。評価はEnsemble.evaluate()またはllm_evaluator.pyを使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

if TYPE_CHECKING:
    from .llm import LLM
    from .sequence_data import SequenceData

from .config import CascadeConfig


class LLMWrapper(nn.Module):
    """
    Hugging Face Trainer用のLLMラッパー。

    SequenceDataのhidden_statesを入力として受け取り、
    loss計算を行うためのラッパー。
    """

    def __init__(self, llm: "LLM"):
        super().__init__()
        self.llm = llm

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with loss computation.

        Args:
            hidden_states: (batch_size, seq_len, dim)
            labels: (batch_size, seq_len)

        Returns:
            Dict with 'loss' and 'logits'
        """
        h_out, _ = self.llm.forward(hidden_states, input_type="hidden_states")
        logits = self.llm.get_logits(h_out)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                reduction='mean'
            )

        return {"loss": loss, "logits": logits}


@dataclass
class SequenceDataCollator:
    """SequenceData用のData Collator。"""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        hidden_states = torch.stack([torch.tensor(f['hidden_states']) for f in features])
        labels = torch.stack([torch.tensor(f['labels']) for f in features])
        return {
            'hidden_states': hidden_states,
            'labels': labels,
        }


class PPLLoggingCallback(TrainerCallback):
    """PPL（パープレキシティ）をログするコールバック。"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.train_ppls: List[float] = []
        self.val_ppls: List[float] = []

    def on_log(  # noqa: ARG002
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        if logs and self.verbose:
            if 'loss' in logs:
                ppl = np.exp(logs['loss'])
                self.train_ppls.append(ppl)
            if 'eval_loss' in logs:
                ppl = np.exp(logs['eval_loss'])
                self.val_ppls.append(ppl)
                print(f"  Epoch {state.epoch:.0f}: val_ppl={ppl:.2f}")


def train_llm(
    llm: "LLM",
    train_data: "SequenceData",
    val_data: "SequenceData",
    training_args: TrainingArguments,
    cascade_config: CascadeConfig,
) -> Tuple["SequenceData", Dict[str, Any]]:
    """
    LLMを訓練し、次のLLM用のhard tokensを返す。

    Hugging Face Trainerを使用した訓練ワークフロー:
    1. Early stopping付きLM訓練（val_dataで検証）
    2. llm.collect_hard_tokens()でhard tokens収集（閾値も設定）

    Args:
        llm: 訓練するLLM
        train_data: 訓練SequenceData (hidden_states, targets)
        val_data: Early stopping用の検証SequenceData
        training_args: Hugging Face TrainingArguments
        cascade_config: CASCADE固有の設定（patience, hard_ratio）

    Returns:
        タプル:
        - SequenceData: 次のLLM用のhard tokens（出力hidden states）
        - Dict: 訓練統計 (train_ppls, val_ppls, best_epoch等)
    """
    verbose = not training_args.disable_tqdm
    batch_size = training_args.per_device_train_batch_size

    if verbose:
        print(f"LLM訓練: {len(train_data)}訓練, {len(val_data)}検証シーケンス")
        print(f"  ({train_data.num_tokens}訓練, {val_data.num_tokens}検証トークン)")

    # 1. Hugging Face Trainerで訓練
    lm_stats = _train_with_hf_trainer(llm, train_data, val_data, training_args, cascade_config)

    # 2. hard tokens収集（llm.thresholdも設定）
    hard_tokens = llm.collect_hard_tokens(train_data, cascade_config.hard_ratio, batch_size)

    # 3. 最終統計を構築
    actual_hard_ratio = hard_tokens.num_tokens / train_data.num_tokens if train_data.num_tokens > 0 else 0.0
    stats: Dict[str, Any] = {
        **lm_stats,
        'hard_ratio': actual_hard_ratio,
        'threshold': llm.threshold,
    }

    if verbose:
        print(f"  閾値 (cos_sim): {llm.threshold:.4f}")
        print(f"  Hard tokens: {len(hard_tokens)}シーケンス ({hard_tokens.num_tokens}トークン, {actual_hard_ratio*100:.1f}%)")

    return hard_tokens, stats


def _train_with_hf_trainer(
    llm: "LLM",
    train_data: "SequenceData",
    val_data: "SequenceData",
    training_args: TrainingArguments,
    cascade_config: CascadeConfig,
) -> Dict[str, Any]:
    """
    Hugging Face Trainerを使用した訓練。

    Args:
        llm: 訓練するLLM
        train_data: 訓練SequenceData
        val_data: 検証SequenceData
        training_args: Hugging Face TrainingArguments
        cascade_config: CASCADE固有の設定

    Returns:
        train_ppls, val_ppls, best_epoch, best_val_ppl, total_epochs, stopped_earlyを含むDict
    """
    device = next(llm.parameters()).device
    verbose = not training_args.disable_tqdm

    # Hugging Face Datasetに変換
    train_dataset = train_data.to_hf_dataset()
    val_dataset = val_data.to_hf_dataset()

    # LLMラッパーを作成
    wrapper = LLMWrapper(llm)
    wrapper.to(device)

    # コールバックを設定
    ppl_callback = PPLLoggingCallback(verbose=verbose)
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=cascade_config.patience),
        ppl_callback,
    ]

    # Trainerを作成
    trainer = Trainer(
        model=wrapper,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=SequenceDataCollator(),
        callbacks=callbacks,
    )

    # 訓練実行
    train_result = trainer.train()

    # 統計を収集
    best_val_ppl = float('inf')
    if ppl_callback.val_ppls:
        best_val_ppl = min(ppl_callback.val_ppls)
        best_epoch = ppl_callback.val_ppls.index(best_val_ppl)
    else:
        best_epoch = 0

    return {
        'train_ppls': ppl_callback.train_ppls,
        'val_ppls': ppl_callback.val_ppls,
        'best_epoch': best_epoch,
        'best_val_ppl': best_val_ppl,
        'total_epochs': int(train_result.metrics.get('epoch', training_args.num_train_epochs)),
        'stopped_early': len(ppl_callback.val_ppls) < int(training_args.num_train_epochs),
    }


def train_llm_simple(
    llm: "LLM",
    train_data: "SequenceData",
    val_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    training_args: TrainingArguments,
    cascade_config: CascadeConfig,
) -> Tuple["SequenceData", Dict[str, Any]]:
    """
    シンプルな訓練ループ（Hugging Face Trainerを使わない場合用）。

    軽量な実装が必要な場合に使用。

    Args:
        llm: 訓練するLLM
        train_data: 訓練SequenceData (hidden_states, targets)
        val_data: Early stopping用の検証SequenceData
        optimizer: LLMパラメータ用オプティマイザ
        training_args: Hugging Face TrainingArguments
        cascade_config: CASCADE固有の設定

    Returns:
        タプル:
        - SequenceData: 次のLLM用のhard tokens（出力hidden states）
        - Dict: 訓練統計 (train_ppls, val_ppls, best_epoch等)
    """
    verbose = not training_args.disable_tqdm
    batch_size = training_args.per_device_train_batch_size

    if verbose:
        print(f"LLM訓練 (simple): {len(train_data)}訓練, {len(val_data)}検証シーケンス")
        print(f"  ({train_data.num_tokens}訓練, {val_data.num_tokens}検証トークン)")

    # 1. Early stopping付きLM訓練
    lm_stats = _train_lm_simple(llm, train_data, val_data, optimizer, training_args, cascade_config)

    # 2. hard tokens収集（llm.thresholdも設定）
    hard_tokens = llm.collect_hard_tokens(train_data, cascade_config.hard_ratio, batch_size)

    # 3. 最終統計を構築
    actual_hard_ratio = hard_tokens.num_tokens / train_data.num_tokens if train_data.num_tokens > 0 else 0.0
    stats: Dict[str, Any] = {
        **lm_stats,
        'hard_ratio': actual_hard_ratio,
        'threshold': llm.threshold,
    }

    if verbose:
        print(f"  閾値 (cos_sim): {llm.threshold:.4f}")
        print(f"  Hard tokens: {len(hard_tokens)}シーケンス ({hard_tokens.num_tokens}トークン, {actual_hard_ratio*100:.1f}%)")

    return hard_tokens, stats


def _train_lm_simple(
    llm: "LLM",
    train_data: "SequenceData",
    val_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    training_args: TrainingArguments,
    cascade_config: CascadeConfig,
) -> Dict[str, Any]:
    """
    シンプルなEarly stopping付き言語モデル訓練。

    Args:
        llm: 訓練するLLM
        train_data: 訓練SequenceData
        val_data: 検証SequenceData
        optimizer: LLMパラメータ用オプティマイザ
        training_args: Hugging Face TrainingArguments
        cascade_config: CASCADE固有の設定

    Returns:
        train_ppls, val_ppls, best_epoch, best_val_ppl, total_epochs, stopped_earlyを含むDict
    """
    device = next(llm.parameters()).device
    verbose = not training_args.disable_tqdm
    max_epochs = int(training_args.num_train_epochs)
    batch_size = training_args.per_device_train_batch_size
    grad_clip = training_args.max_grad_norm

    best_ppl = float('inf')
    best_state: Dict[str, torch.Tensor] | None = None
    patience_counter = 0
    best_epoch = 0
    epoch = 0

    train_ppls: List[float] = []
    val_ppls: List[float] = []

    for epoch in range(max_epochs):
        # 訓練エポック
        train_ppl = _run_train_epoch(llm, train_data, optimizer, batch_size, grad_clip)
        train_ppls.append(train_ppl)

        # 検証PPL計算（early stopping用）
        from .llm_evaluator import compute_ppl
        val_ppl = compute_ppl(llm, val_data, batch_size)
        val_ppls.append(val_ppl)

        # Early stoppingチェック
        is_best = val_ppl < best_ppl
        if is_best:
            best_ppl = val_ppl
            best_state = {k: v.cpu().clone() for k, v in llm.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if verbose:
            status = "best" if is_best else f"{patience_counter}/{cascade_config.patience}"
            print(f"  Epoch {epoch+1}/{max_epochs}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f} [{status}]")

        if patience_counter >= cascade_config.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # ベストモデルを復元
    if best_state is not None:
        llm.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {
        'train_ppls': train_ppls,
        'val_ppls': val_ppls,
        'best_epoch': best_epoch,
        'best_val_ppl': best_ppl,
        'total_epochs': epoch + 1,
        'stopped_early': patience_counter >= cascade_config.patience,
    }


def _run_train_epoch(
    llm: "LLM",
    train_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    grad_clip: float,
) -> float:
    """
    【訓練用】1訓練エポックを実行。

    勾配計算とパラメータ更新を行う。

    Args:
        llm: 訓練するLLM
        train_data: 訓練SequenceData
        optimizer: LLMパラメータ用オプティマイザ
        batch_size: バッチサイズ
        grad_clip: 勾配クリッピング値

    Returns:
        このエポックの訓練パープレキシティ
    """
    device = next(llm.parameters()).device
    llm.train()
    total_loss = 0.0
    total_tokens = 0

    for h, y in train_data.to(str(device)).batches(batch_size, shuffle=True):
        optimizer.zero_grad()
        h_out, _ = llm.forward(h, input_type="hidden_states")
        logits = llm.get_logits(h_out)

        batch_size_actual, seq_len, vocab_size = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            reduction='sum'
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_tokens += batch_size_actual * seq_len

    return float(np.exp(total_loss / total_tokens))
