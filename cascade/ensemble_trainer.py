"""
CASCADEフレームワーク - Ensemble訓練

完全なEnsemble（全LLM）の訓練関数。
"""

from __future__ import annotations

import torch
from typing import Dict, Any, List, Tuple, TYPE_CHECKING
from dataclasses import replace

from transformers import TrainingArguments

if TYPE_CHECKING:
    from .ensemble import Ensemble
    from .sequence_data import SequenceData

from .llm_trainer import train_llm
from .config import CascadeConfig
from .sequence_data import SequenceData


def train_ensemble(
    ensemble: "Ensemble",
    train_data: "SequenceData",
    val_data: "SequenceData",
    training_args: TrainingArguments,
    cascade_config: CascadeConfig,
) -> Dict[str, Any]:
    """
    Ensembleの全LLMを順番に訓練。

    訓練フロー:
    1. LLM 0を全訓練データで訓練 → hard tokens収集
    2. LLM 1をhard tokensで訓練 → hard tokens収集
    3. ... 全LLMまで続行

    Args:
        ensemble: 訓練するEnsemble
        train_data: 訓練SequenceData（埋め込み済みトークンをhidden statesとして）
        val_data: Early stopping用の検証SequenceData
        training_args: Hugging Face TrainingArguments
        cascade_config: CASCADE固有の設定（patience, hard_ratio, lr_decay）

    Returns:
        LLMごとの統計と全体の訓練情報を含むDict
    """
    verbose = not training_args.disable_tqdm
    all_stats: List[Dict[str, Any]] = []
    current_train_data = train_data
    current_val_data = val_data

    for llm_idx, llm in enumerate(ensemble.llms):
        is_last_llm = (llm_idx == len(ensemble.llms) - 1)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"LLM {llm_idx} 訓練")
            print("=" * 60)

        if len(current_train_data) == 0:
            if verbose:
                print(f"LLM {llm_idx}用のデータなし - スキップ")
            break

        # 深いLLMほど学習率を減衰
        llm_lr = training_args.learning_rate * (cascade_config.lr_decay ** llm_idx)
        llm_training_args = replace(
            training_args,
            learning_rate=llm_lr,
            output_dir=f"{training_args.output_dir}/llm_{llm_idx}",
        )

        hard_data, stats = train_llm(
            llm, current_train_data, current_val_data, llm_training_args, cascade_config
        )

        all_stats.append({
            'llm_idx': llm_idx,
            'lr': llm_lr,
            **stats,
        })

        if verbose:
            print(f"\nLLM {llm_idx} 結果:")
            print(f"  Best PPL: {stats['best_val_ppl']:.2f}")
            print(f"  閾値: {stats['threshold']:.4f}")
            print(f"  Hard tokens: {len(hard_data)}シーケンス ({hard_data.num_tokens}トークン)")

        # 次のLLM用にデータを更新（最終LLM以外）
        if not is_last_llm:
            current_train_data = hard_data
            # val_dataを訓練済みLLMで変換
            batch_size = training_args.per_device_train_batch_size
            current_val_data = llm.transform_data(current_val_data, batch_size)

    return {
        'llm_stats': all_stats,
        'num_llms_trained': len(all_stats),
    }


def create_sequence_data(
    ensemble: "Ensemble",
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> "SequenceData":
    """
    トークンを埋め込んでSequenceDataを作成。

    最初のLLMのembeddingを使用してtoken_idsをhidden statesに変換。

    Args:
        ensemble: Ensemble（最初のLLMのembedding使用）
        batches: (x, y)バッチのリスト

    Returns:
        埋め込み済みhidden statesとtargetsを持つSequenceData
    """
    device = next(ensemble.parameters()).device
    first_llm = ensemble.llms[0]

    all_hidden: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            # CausalLMのembeddingを使用
            # GPT2の場合: base_llm.transformer.wte
            # 一般的なHugging Faceモデルの場合: get_input_embeddings()
            embedding = first_llm.base_llm.get_input_embeddings()
            h = embedding(x)
            all_hidden.append(h)
            all_targets.append(y)

    return SequenceData(
        torch.cat(all_hidden),
        torch.cat(all_targets),
    )
