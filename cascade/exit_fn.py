"""
CASCADEフレームワーク - Exit関数

hidden_historyに基づいて早期exitを判定する関数群。
デフォルト実装はCALM式のcosine similarityを使用。

重要: 最終レイヤー（hidden_history[-1]）は出力変換が入っているため、
cos_sim計算には使用しない。代わりに[-2]（最終変換前）を使用する。
"""

import torch
import torch.nn.functional as F
from typing import List, Callable, Sequence, Union


# Exit関数の型エイリアス
ExitFn = Callable[[List[torch.Tensor], float], torch.Tensor]

# hidden_historyの型（リストまたはタプル）
HiddenHistory = Union[List[torch.Tensor], Sequence[torch.Tensor]]


def compute_cos_sim_from_history(hidden_history: HiddenHistory) -> torch.Tensor:
    """
    hidden_historyからcos_simを計算。

    レイヤー数に応じて適切なcos_sim計算を行う:
    - 4層以上: 最後から2-3番目のレイヤーの平均cos_sim
    - 2-3層: 最後から1-2番目のcos_sim
    - 1層: 入力と出力のcos_sim

    重要: 最終レイヤー（hidden_history[-1]）は出力変換が入っているため使用しない。

    Args:
        hidden_history: hidden statesのリスト/タプル [入力, layer1出力, ..., layerN出力]
                       (N+1) 要素

    Returns:
        cos_sim: トークンごとのcosine similarity (batch_size, seq_len)
    """
    num_states = len(hidden_history)

    if num_states >= 5:
        # 4層以上: 最後から2-3番目のレイヤーの平均
        h_prev2 = hidden_history[-4]
        h_prev1 = hidden_history[-3]
        h_last = hidden_history[-2]
        cos_sim_1 = compute_cos_sim(h_prev2, h_prev1)
        cos_sim_2 = compute_cos_sim(h_prev1, h_last)
        return (cos_sim_1 + cos_sim_2) / 2.0
    elif num_states >= 3:
        # 2-3層: 最後から1-2番目のcos_sim
        h_prev = hidden_history[-3]
        h_last = hidden_history[-2]
        return compute_cos_sim(h_prev, h_last)
    else:
        # 1層: 入力と出力のcos_sim
        h_in = hidden_history[0]
        h_out = hidden_history[-2] if num_states > 1 else hidden_history[-1]
        return compute_cos_sim(h_in, h_out)


def default_exit_fn(hidden_history: List[torch.Tensor], threshold: float) -> torch.Tensor:
    """
    レイヤー数に応じた適応的なexit関数。

    compute_cos_sim_from_history()を使用してcos_simを計算し、
    閾値と比較してexit判定を行う。

    Args:
        hidden_history: hidden statesのリスト [入力, レイヤー1出力, ...]
        threshold: exitのためのcosine similarity閾値

    Returns:
        should_exit: Booleanマスク (batch_size, seq_len) True = exitすべき
    """
    cos_sim = compute_cos_sim_from_history(hidden_history)
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
