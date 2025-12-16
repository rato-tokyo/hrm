"""
CASCADEフレームワーク - Exit関数

hidden_historyに基づいて早期exitを判定する関数群。
デフォルト実装はCALM式のcosine similarityを使用。
"""

import torch
import torch.nn.functional as F
from typing import List, Callable


# Exit関数の型エイリアス
ExitFn = Callable[[List[torch.Tensor], float], torch.Tensor]


def default_exit_fn(hidden_history: List[torch.Tensor], threshold: float) -> torch.Tensor:
    """
    Layer 28-29の平均cos_simを使用するexit関数。

    Layer 30（最終レイヤー）は出力変換で全トークンが大きく変化するため、
    exit判定には不適切。Layer 28-29の変化量の平均を使用する。

    注: hidden_historyは [embedding, layer1出力, ..., layer30出力] の31要素。
        Layer 29の変化 = hidden_history[-2] vs hidden_history[-1] ではなく
        hidden_history[29] vs hidden_history[30] = index 29 vs 30

    Args:
        hidden_history: hidden statesのリスト [入力, レイヤー1出力, ...]
        threshold: exitのためのcosine similarity閾値

    Returns:
        should_exit: Booleanマスク (batch_size, seq_len) True = exitすべき
    """
    # Layer 29の変化（Layer 28出力 → Layer 29出力）
    # hidden_history[28] = Layer 28出力, hidden_history[29] = Layer 29出力
    h_28 = hidden_history[-3]  # Layer 28出力（=Layer 29入力）
    h_29 = hidden_history[-2]  # Layer 29出力（=Layer 30入力）

    # Layer 28の変化（Layer 27出力 → Layer 28出力）
    h_27 = hidden_history[-4]  # Layer 27出力（=Layer 28入力）

    # 各レイヤーのcos_sim計算
    cos_sim_28 = compute_cos_sim(h_27, h_28)  # Layer 28での変化
    cos_sim_29 = compute_cos_sim(h_28, h_29)  # Layer 29での変化

    # 平均を取る
    avg_cos_sim = (cos_sim_28 + cos_sim_29) / 2.0

    return avg_cos_sim >= threshold


def compute_cos_sim(h_in: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
    """
    2つのhidden states間のcosine similarityを計算。

    Args:
        h_in: 入力hidden states (batch_size, seq_len, dim)
        h_out: 出力hidden states (batch_size, seq_len, dim)

    Returns:
        cos_sim: トークンごとのcosine similarity (batch_size, seq_len)
    """
    h_in_norm = F.normalize(h_in, dim=-1)
    h_out_norm = F.normalize(h_out, dim=-1)
    return (h_in_norm * h_out_norm).sum(dim=-1)
