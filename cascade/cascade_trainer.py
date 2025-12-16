"""
CASCADEフレームワーク - CascadeTrainer

HF Trainerを活用したCASCADE訓練クラス。
各LLMを順番にHF Trainerで訓練し、hard tokensを次のLLMに渡す。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, replace

from datasets import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

from .config import CascadeConfig
from .cascade_dataset import (
    create_cascade_dataset,
    get_dataset_info,
    collect_hard_tokens_from_dataset,
    transform_dataset,
)

if TYPE_CHECKING:
    from .llm import LLM
    from .ensemble import Ensemble


class LLMWrapper(nn.Module):
    """
    Hugging Face Trainer用のLLMラッパー。

    hidden_statesを入力として受け取り、loss計算を行う。
    """

    def __init__(self, llm: "LLM"):
        super().__init__()
        self.llm = llm

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass with loss computation."""
        # dtype変換はLLM.forwardで自動実行
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
class CascadeDataCollator:
    """Cascade Dataset用のData Collator。"""

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
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs  # noqa: ARG002
    ):
        del args, control, kwargs  # unused but required by TrainerCallback interface
        if logs and self.verbose:
            if 'loss' in logs:
                ppl = np.exp(logs['loss'])
                self.train_ppls.append(ppl)
            if 'eval_loss' in logs:
                ppl = np.exp(logs['eval_loss'])
                self.val_ppls.append(ppl)
                print(f"  Epoch {state.epoch:.0f}: val_ppl={ppl:.2f}")


class CascadeTrainer:
    """
    HF Trainerを活用したCASCADE訓練クラス。

    各LLMを順番にHF Trainerで訓練し、hard tokensを次のLLMに渡す。

    使用例:
        cascade_trainer = CascadeTrainer(training_args, cascade_config)
        stats = cascade_trainer.train(ensemble, train_dataset, val_dataset)
    """

    def __init__(
        self,
        training_args: TrainingArguments,
        cascade_config: CascadeConfig,
    ):
        """
        Args:
            training_args: Hugging Face TrainingArguments
            cascade_config: CASCADE固有の設定
        """
        self.training_args = training_args
        self.cascade_config = cascade_config

    def train(
        self,
        ensemble: "Ensemble",
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> Dict[str, Any]:
        """
        Ensembleの全LLMを順番に訓練。

        Args:
            ensemble: 訓練するEnsemble
            train_dataset: 訓練用HF Dataset (hidden_states, labels)
            val_dataset: 検証用HF Dataset

        Returns:
            LLMごとの統計と全体の訓練情報を含むDict
        """
        verbose = not self.training_args.disable_tqdm
        all_stats: List[Dict[str, Any]] = []
        current_train = train_dataset
        current_val = val_dataset

        for llm_idx, llm in enumerate(ensemble.llms):
            is_last_llm = (llm_idx == len(ensemble.llms) - 1)

            # フリーズされたLLMかどうかを確認
            trainable_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
            is_frozen = (trainable_params == 0)

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"LLM {llm_idx} {'(フリーズ済み - 訓練スキップ)' if is_frozen else '訓練'}")
                print("=" * 60)

            train_info = get_dataset_info(current_train)
            if train_info['num_sequences'] == 0:
                if verbose:
                    print(f"LLM {llm_idx}用のデータなし - スキップ")
                break

            # フリーズされたLLMは訓練せずhard token収集のみ
            if is_frozen:
                if verbose:
                    print("  訓練可能パラメータ: 0 - 訓練スキップ")

                # hard tokens収集
                batch_size = self.training_args.per_device_train_batch_size
                hard_dataset, threshold = collect_hard_tokens_from_dataset(
                    llm, current_train, self.cascade_config.hard_ratio, batch_size
                )
                llm.threshold = threshold

                hard_info = get_dataset_info(hard_dataset)
                actual_hard_ratio = hard_info['num_tokens'] / train_info['num_tokens'] if train_info['num_tokens'] > 0 else 0.0

                all_stats.append({
                    'llm_idx': llm_idx,
                    'lr': 0.0,
                    'best_val_ppl': float('nan'),
                    'hard_ratio': actual_hard_ratio,
                    'threshold': threshold,
                    'frozen': True,
                })

                if verbose:
                    print(f"  閾値: {threshold:.4f}")
                    print(f"  Hard tokens: {hard_info['num_sequences']}シーケンス ({hard_info['num_tokens']}トークン)")

                # 次のLLM用にデータを更新
                if not is_last_llm:
                    current_train = hard_dataset
                    current_val = transform_dataset(llm, current_val, batch_size)
                continue

            # 深いLLMほど学習率を減衰
            llm_lr = self.training_args.learning_rate * (self.cascade_config.lr_decay ** llm_idx)
            # モデルがfloat16の場合、AMPを無効化（float16勾配のunscaleエラー回避）
            model_dtype = next(llm.parameters()).dtype
            use_fp16 = model_dtype != torch.float16
            llm_training_args = replace(
                self.training_args,
                learning_rate=llm_lr,
                output_dir=f"{self.training_args.output_dir}/llm_{llm_idx}",
                fp16=use_fp16 and self.training_args.fp16,
                bf16=False,  # bf16も無効化
            )

            # HF Trainerで訓練
            train_stats = self._train_single_llm(llm, current_train, current_val, llm_training_args)

            # hard tokens収集
            batch_size = self.training_args.per_device_train_batch_size
            hard_dataset, threshold = collect_hard_tokens_from_dataset(
                llm, current_train, self.cascade_config.hard_ratio, batch_size
            )
            llm.threshold = threshold

            hard_info = get_dataset_info(hard_dataset)
            actual_hard_ratio = hard_info['num_tokens'] / train_info['num_tokens'] if train_info['num_tokens'] > 0 else 0.0

            all_stats.append({
                'llm_idx': llm_idx,
                'lr': llm_lr,
                **train_stats,
                'hard_ratio': actual_hard_ratio,
                'threshold': threshold,
            })

            if verbose:
                print(f"\nLLM {llm_idx} 結果:")
                print(f"  Best PPL: {train_stats['best_val_ppl']:.2f}")
                print(f"  閾値: {threshold:.4f}")
                print(f"  Hard tokens: {hard_info['num_sequences']}シーケンス ({hard_info['num_tokens']}トークン)")

            # 次のLLM用にデータを更新（最終LLM以外）
            if not is_last_llm:
                current_train = hard_dataset
                # val_dataを訓練済みLLMで変換
                current_val = transform_dataset(llm, current_val, batch_size)

        return {
            'llm_stats': all_stats,
            'num_llms_trained': len(all_stats),
        }

    def _train_single_llm(
        self,
        llm: "LLM",
        train_dataset: Dataset,
        val_dataset: Dataset,
        training_args: TrainingArguments,
    ) -> Dict[str, Any]:
        """
        単一のLLMをHF Trainerで訓練。

        Args:
            llm: 訓練するLLM
            train_dataset: 訓練用Dataset
            val_dataset: 検証用Dataset
            training_args: TrainingArguments

        Returns:
            訓練統計を含むDict
        """
        device = next(llm.parameters()).device
        verbose = not training_args.disable_tqdm

        train_info = get_dataset_info(train_dataset)
        val_info = get_dataset_info(val_dataset)

        if verbose:
            print(f"LLM訓練: {train_info['num_sequences']}訓練, {val_info['num_sequences']}検証シーケンス")
            print(f"  ({train_info['num_tokens']}訓練, {val_info['num_tokens']}検証トークン)")

        # LLMラッパーを作成
        wrapper = LLMWrapper(llm)
        wrapper.to(device)

        # コールバックを設定
        ppl_callback = PPLLoggingCallback(verbose=verbose)
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=self.cascade_config.patience),
            ppl_callback,
        ]

        # Trainerを作成
        trainer = Trainer(
            model=wrapper,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=CascadeDataCollator(),
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


def create_initial_dataset(
    ensemble: "Ensemble",
    batches: List[tuple[torch.Tensor, torch.Tensor]],
) -> Dataset:
    """
    トークンを埋め込んでDatasetを作成。

    最初のLLMのembeddingを使用してtoken_idsをhidden statesに変換。

    Args:
        ensemble: Ensemble（最初のLLMのembedding使用）
        batches: (x, y)バッチのリスト

    Returns:
        埋め込み済みhidden statesとlabelsを持つDataset
    """
    device = next(ensemble.parameters()).device
    first_llm = ensemble.llms[0]

    all_hidden: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            embedding = first_llm.base_llm.get_input_embeddings()
            h = embedding(x)
            all_hidden.append(h.cpu())
            all_targets.append(y.cpu())

    return create_cascade_dataset(torch.cat(all_hidden), torch.cat(all_targets))
