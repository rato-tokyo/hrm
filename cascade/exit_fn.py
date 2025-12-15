"""
LEGOフレームワーク - Exit関数

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
    CALM式のデフォルトexit関数（cosine similarity使用）。

    最後の2つのhidden states（最終レイヤーの入出力）を比較。
    類似度が高い = 変化が小さい = 収束したと判断。

    Args:
        hidden_history: hidden statesのリスト [入力, レイヤー1出力, ...]
        threshold: exitのためのcosine similarity閾値

    Returns:
        should_exit: Booleanマスク (batch_size, seq_len) True = exitすべき
    """
    h_in = hidden_history[-2]   # 最終レイヤーへの入力
    h_out = hidden_history[-1]  # 最終レイヤーの出力

    # Cosine similarity
    h_in_norm = F.normalize(h_in, dim=-1)
    h_out_norm = F.normalize(h_out, dim=-1)
    cos_sim = (h_in_norm * h_out_norm).sum(dim=-1)

    return cos_sim >= threshold


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
