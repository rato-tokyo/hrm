"""
LEGOフレームワーク - モデル評価（評価専用）

TRUE Early ExitによるLEGOEnsemble評価関数。

注意: このモジュールは評価専用。訓練はensemble_trainer.pyを使用。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .lego_ensemble import LEGOEnsemble


def evaluate_ensemble(
    ensemble: "LEGOEnsemble",
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """
    【評価用】TRUE Early ExitでLEGOEnsembleを評価。

    訓練完了後のモデル評価に使用。PPL、Accuracy、exit統計を計算。

    Args:
        ensemble: 訓練済みLEGOEnsemble
        val_batches: 検証用の(x, y)バッチのリスト

    Returns:
        ppl, accuracy, shallow_ratio, compute_cost, exit_countsを含むDict
    """
    device = next(ensemble.parameters()).device
    ensemble.eval()

    total_loss = 0.0
    total_tokens = 0
    correct = 0
    aggregated_exit_counts: List[int] = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            logits, stats = ensemble.evaluate(x)

            # exit countsを集計
            batch_exit_counts: List[int] = stats['exit_counts']
            if not aggregated_exit_counts:
                aggregated_exit_counts = [0] * len(batch_exit_counts)
            for i, count in enumerate(batch_exit_counts):
                aggregated_exit_counts[i] += count

            # LossとAccuracy
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            total_loss += F.cross_entropy(logits_flat, y_flat, reduction='sum').item()
            total_tokens += y_flat.numel()
            correct += int((logits_flat.argmax(dim=-1) == y_flat).sum().item())

    ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
    acc = correct / total_tokens if total_tokens > 0 else 0.0

    # モデルのメソッドで統計を計算
    exit_stats = ensemble._compute_exit_stats(aggregated_exit_counts)

    return {
        'ppl': ppl,
        'accuracy': acc,
        'shallow_ratio': exit_stats['shallow_ratio'],
        'compute_cost': exit_stats['compute_cost'],
        'compute_savings': 1.0 - exit_stats['compute_cost'],
        'exit_counts': aggregated_exit_counts,
        'total_tokens': total_tokens,
    }
