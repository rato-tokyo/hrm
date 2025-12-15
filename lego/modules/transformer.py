"""
LEGOフレームワーク - Transformerコンポーネント

注意: 本フレームワークは事前学習専用です。KVキャッシュは実装していません。
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .norm import RMSNorm
from .attention import MultiHeadAttention
from .ffn import GatedLinearUnit


class TransformerLayer(nn.Module):
    """
    単一のTransformerレイヤー（Attention + FFN）。

    Post-Normアーキテクチャ。事前学習専用（KVキャッシュなし）。
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, max_seq_len: int, causal: bool, eps: float):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, max_seq_len, causal)
        self.ffn = GatedLinearUnit(dim, ffn_dim)
        self.norm1 = RMSNorm(dim, eps)
        self.norm2 = RMSNorm(dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-Normアーキテクチャ
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerBlock(nn.Module):
    """
    TransformerLayerのスタック。

    単独で使用することも、LEGOBlockでラップして早期exit機能を追加することも可能。

    Args:
        dim: モデル次元
        num_heads: Attentionヘッド数
        num_layers: このブロック内のレイヤー数
        ffn_dim: FFN隠れ層次元
        max_seq_len: 最大シーケンス長
        causal: Causalマスクを使用するか
        eps: RMSNormのイプシロン
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int,
        causal: bool,
        eps: float
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, ffn_dim, max_seq_len, causal, eps) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        全レイヤーを通過し、hidden historyを返す。

        Args:
            x: 入力テンソル (batch_size, seq_len, dim)

        Returns:
            タプル:
            - 出力テンソル (batch_size, seq_len, dim)
            - hidden history: 各hidden stateのリスト [入力, レイヤー1出力, レイヤー2出力, ...]
        """
        hidden_history = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_history.append(x)
        return x, hidden_history
