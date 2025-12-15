"""
LEGOフレームワーク - LEGOEnsemble訓練

完全なLEGOEnsemble（全LLM）の訓練関数。
"""

from __future__ import annotations

import torch
from typing import Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .lego_ensemble import LEGOEnsemble
    from .sequence_data import SequenceData

from .llm_trainer import train_llm
from .config import TrainerConfig
from .sequence_data import SequenceData


def train_ensemble(
    ensemble: "LEGOEnsemble",
    train_data: "SequenceData",
    val_data: "SequenceData",
    config: TrainerConfig,
    lr_decay: float,
) -> Dict[str, Any]:
    """
    LEGOEnsembleの全LLMを順番に訓練。

    訓練フロー:
    1. LLM 0（既存LLM）を全訓練データで訓練 → hard tokens収集
    2. LLM 1（未学習）をhard tokensで訓練 → hard tokens収集
    3. ... 全LLMまで続行

    Args:
        ensemble: 訓練するLEGOEnsemble
        train_data: 訓練SequenceData（埋め込み済みトークンをhidden statesとして）
        val_data: Early stopping用の検証SequenceData
        config: 訓練用TrainerConfig
        lr_decay: 後続LLMごとの学習率減衰係数

    Returns:
        LLMごとの統計と全体の訓練情報を含むDict
    """
    all_stats: List[Dict[str, Any]] = []
    current_train_data = train_data
    current_val_data = val_data

    for llm_idx, llm in enumerate(ensemble.llms):
        is_last_llm = (llm_idx == len(ensemble.llms) - 1)

        if config.verbose:
            print(f"\n{'=' * 60}")
            print(f"LLM {llm_idx} 訓練")
            print("=" * 60)

        if len(current_train_data) == 0:
            if config.verbose:
                print(f"LLM {llm_idx}用のデータなし - スキップ")
            break

        # 深いLLMほど学習率を減衰
        llm_lr = config.lr * (lr_decay ** llm_idx)
        llm_config = TrainerConfig(
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            patience=config.patience,
            grad_clip=config.grad_clip,
            hard_ratio=config.hard_ratio,
            lr=llm_lr,
            verbose=config.verbose,
        )

        optimizer = torch.optim.AdamW(llm.parameters(), lr=llm_lr)
        hard_data, stats = train_llm(
            llm, current_train_data, current_val_data, optimizer, llm_config
        )

        all_stats.append({
            'llm_idx': llm_idx,
            'lr': llm_lr,
            **stats,
        })

        if config.verbose:
            print(f"\nLLM {llm_idx} 結果:")
            print(f"  Best PPL: {stats['best_val_ppl']:.2f}")
            print(f"  閾値: {stats['threshold']:.4f}")
            print(f"  Hard tokens: {len(hard_data)}シーケンス ({hard_data.num_tokens}トークン)")

        # 次のLLM用にデータを更新（最終LLM以外）
        if not is_last_llm:
            current_train_data = hard_data
            # val_dataを訓練済みLLMで変換
            current_val_data = llm.transform_data(current_val_data, config.batch_size)

    return {
        'llm_stats': all_stats,
        'num_llms_trained': len(all_stats),
    }


def create_sequence_data(
    ensemble: "LEGOEnsemble",
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> "SequenceData":
    """
    トークンを埋め込んでSequenceDataを作成。

    Args:
        ensemble: LEGOEnsemble（embeddingレイヤー用）
        batches: (x, y)バッチのリスト

    Returns:
        埋め込み済みhidden statesとtargetsを持つSequenceData
    """
    device = next(ensemble.parameters()).device

    all_hidden: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            h = ensemble.embedding(x)
            all_hidden.append(h)
            all_targets.append(y)

    return SequenceData(
        torch.cat(all_hidden),
        torch.cat(all_targets),
    )
