"""
Dual-Context Attention (DCA)

L0 (Local Context) と L1 (Representative Context) の
2層コンテキストを統合するアテンション機構。

アーキテクチャ:
- L0: 最新のhidden states（通常のKVキャッシュ相当）
- L1: 過去spanの代表ベクトル（双方向エンコーダで圧縮）

L1の代表ベクトルはKeyのみを持ち、Query@Keyでattentionを計算後、
代表ベクトル自体をValueとして使用する。

Reference:
- Infini-attention: Efficient Infinite Context Transformers
- Memorizing Transformers (ICLR 2022)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field


class DCAOutput(NamedTuple):
    """Dual-Context Attentionの出力。"""
    output: Tensor  # (batch, seq_len, dim)
    l0_attention: Tensor  # L0へのattention weights
    l1_attention: Optional[Tensor]  # L1へのattention weights


@dataclass
class DualContextState:
    """Dual-Contextの状態。"""
    # L0: Local Context (最新のhidden states)
    l0_keys: Optional[Tensor] = None    # (batch, l0_len, dim)
    l0_values: Optional[Tensor] = None  # (batch, l0_len, dim)

    # L1: Representative Context (圧縮された代表ベクトル)
    l1_representatives: List[Tensor] = field(default_factory=list)  # List of (dim,)
    l1_keys: Optional[Tensor] = None    # (batch, l1_len, dim) - キャッシュ

    # メタデータ
    l0_positions: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) pairs
    l1_positions: List[Tuple[int, int]] = field(default_factory=list)  # 圧縮元のspan位置
    total_processed: int = 0


class DualContextMemory(nn.Module):
    """
    L0/L1の2層コンテキストを管理するメモリ。

    L0: 最新のwindow_size分のhidden states（KVキャッシュ）
    L1: 古いspanの代表ベクトル（双方向エンコーダで圧縮）

    使用例:
        memory = DualContextMemory(dim=768, window_size=512, max_representatives=256)

        for hidden_chunk in stream:
            state = memory.update(hidden_chunk)
            # state.l0_keys, state.l0_values: 最新コンテキスト
            # state.l1_keys: 代表ベクトルから生成したkeys
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 512,
        max_representatives: int = 256,
        encoder_mode: str = "bilstm",
        encoder_layers: int = 1,
        encoder_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: 隠れ状態の次元
            window_size: L0のウィンドウサイズ
            max_representatives: L1の最大代表ベクトル数
            encoder_mode: 双方向エンコーダのモード
            encoder_layers: エンコーダのレイヤー数
            encoder_heads: Transformerのヘッド数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.max_representatives = max_representatives

        # L0用のKey/Value投影
        self.l0_key_proj = nn.Linear(dim, dim)
        self.l0_value_proj = nn.Linear(dim, dim)

        # L1用のKey投影（代表ベクトル → Key）
        self.l1_key_proj = nn.Linear(dim, dim)

        # 双方向エンコーダ（span → 代表ベクトル）
        from .infini_span_memory import BidirectionalSpanEncoder
        self.span_encoder = BidirectionalSpanEncoder(
            dim=dim,
            mode=encoder_mode,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            dropout=dropout,
        )

        # 状態
        self._state = DualContextState()
        self._l0_buffer: List[Tensor] = []  # 未処理のhidden states
        self._l0_buffer_positions: List[int] = []

    def reset(self):
        """メモリをリセット。"""
        self._state = DualContextState()
        self._l0_buffer = []
        self._l0_buffer_positions = []

    @property
    def state(self) -> DualContextState:
        """現在の状態。"""
        return self._state

    def update(
        self,
        hidden_states: Tensor,
        force_compress: bool = False,
    ) -> DualContextState:
        """
        新しいhidden statesでメモリを更新。

        Args:
            hidden_states: (seq_len, dim) or (batch, seq_len, dim)
            force_compress: L0がwindow_sizeに達していなくても圧縮

        Returns:
            更新された状態
        """
        # バッチ次元を正規化
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        batch_size, seq_len, _ = hidden_states.shape

        # バッファに追加
        for t in range(seq_len):
            self._l0_buffer.append(hidden_states[:, t:t+1, :])
            self._l0_buffer_positions.append(self._state.total_processed + t)

        self._state.total_processed += seq_len

        # L0がwindow_sizeを超えたら古い部分をL1に移動
        while len(self._l0_buffer) > self.window_size or force_compress:
            if len(self._l0_buffer) <= self.window_size // 2:
                break

            # 前半をspan として圧縮
            compress_size = len(self._l0_buffer) - self.window_size // 2
            compress_size = min(compress_size, self.window_size // 2)

            if compress_size < 1:
                break

            # 圧縮対象を取り出し
            span_hidden = torch.cat(self._l0_buffer[:compress_size], dim=1)  # (batch, compress_size, dim)
            span_start = self._l0_buffer_positions[0]
            span_end = self._l0_buffer_positions[compress_size - 1]

            # バッファから削除
            self._l0_buffer = self._l0_buffer[compress_size:]
            self._l0_buffer_positions = self._l0_buffer_positions[compress_size:]

            # 双方向エンコーダで圧縮
            representative = self.span_encoder(span_hidden.squeeze(0))  # (dim,)

            # L1に追加
            self._state.l1_representatives.append(representative.detach())
            self._state.l1_positions.append((span_start, span_end))

            # 最大数を超えたら古いものを削除
            if len(self._state.l1_representatives) > self.max_representatives:
                self._state.l1_representatives.pop(0)
                self._state.l1_positions.pop(0)

        # L0のKV計算
        if self._l0_buffer:
            l0_hidden = torch.cat(self._l0_buffer, dim=1)  # (batch, l0_len, dim)
            self._state.l0_keys = self.l0_key_proj(l0_hidden)
            self._state.l0_values = self.l0_value_proj(l0_hidden)
            self._state.l0_positions = [
                (self._l0_buffer_positions[0], self._l0_buffer_positions[-1])
            ]
        else:
            self._state.l0_keys = None
            self._state.l0_values = None
            self._state.l0_positions = []

        # L1のKey計算
        if self._state.l1_representatives:
            l1_reps = torch.stack(self._state.l1_representatives, dim=0)  # (l1_len, dim)
            l1_reps = l1_reps.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, l1_len, dim)
            self._state.l1_keys = self.l1_key_proj(l1_reps)
        else:
            self._state.l1_keys = None

        return self._state

    def get_context_summary(self) -> dict:
        """コンテキストの要約情報。"""
        l0_len = len(self._l0_buffer)
        l1_len = len(self._state.l1_representatives)

        return {
            "l0_length": l0_len,
            "l1_length": l1_len,
            "total_processed": self._state.total_processed,
            "compression_ratio": 1.0 - (l0_len + l1_len) / max(self._state.total_processed, 1),
        }


class DualContextAttention(nn.Module):
    """
    L0とL1を統合するDual-Context Attention。

    アルゴリズム:
    1. Query @ L0_Keys → L0 attention weights
    2. Query @ L1_Keys → L1 attention weights
    3. Softmax([L0_weights, L1_weights])で正規化
    4. L0: weights @ L0_Values
    5. L1: weights @ L1_Representatives（代表ベクトル自体がValue）
    6. Gate-based mixing or Concatenation

    使用例:
        attention = DualContextAttention(dim=768, num_heads=8)

        output = attention(
            query=current_hidden,
            l0_keys=state.l0_keys,
            l0_values=state.l0_values,
            l1_keys=state.l1_keys,
            l1_values=l1_representatives,  # 代表ベクトル自体
        )
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init: float = 0.0,
        combine_mode: str = "gate",
    ):
        """
        Args:
            dim: 隠れ状態の次元
            num_heads: Attentionヘッド数
            dropout: ドロップアウト率
            gate_init: L0/L1混合ゲートの初期値
            combine_mode: "gate" or "concat"
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.combine_mode = combine_mode

        # Query投影
        self.q_proj = nn.Linear(dim, dim)

        # Output投影
        if combine_mode == "concat":
            self.out_proj = nn.Linear(dim * 2, dim)
        else:
            self.out_proj = nn.Linear(dim, dim)

        # Gate（gateモード時）
        self.gate = nn.Parameter(torch.tensor(gate_init))

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: Tensor,
        l0_keys: Optional[Tensor] = None,
        l0_values: Optional[Tensor] = None,
        l1_keys: Optional[Tensor] = None,
        l1_values: Optional[Tensor] = None,
        l0_mask: Optional[Tensor] = None,
    ) -> DCAOutput:
        """
        Dual-Context Attentionを実行。

        Args:
            query: (batch, seq_len, dim)
            l0_keys: (batch, l0_len, dim) - L0のKey
            l0_values: (batch, l0_len, dim) - L0のValue
            l1_keys: (batch, l1_len, dim) - L1のKey
            l1_values: (batch, l1_len, dim) - L1のValue（代表ベクトル）
            l0_mask: (seq_len, l0_len) - L0へのCausal mask

        Returns:
            DCAOutput
        """
        batch_size, seq_len, _ = query.shape

        # Query投影
        q = self.q_proj(query)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (batch, num_heads, seq_len, head_dim)

        # === L0 Attention ===
        if l0_keys is not None and l0_values is not None:
            l0_len = l0_keys.size(1)

            k0 = l0_keys.view(batch_size, l0_len, self.num_heads, self.head_dim).transpose(1, 2)
            v0 = l0_values.view(batch_size, l0_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            attn_l0 = torch.matmul(q, k0.transpose(-2, -1)) * self.scale

            # Causal mask
            if l0_mask is not None:
                attn_l0 = attn_l0.masked_fill(l0_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

            attn_l0_weights = F.softmax(attn_l0, dim=-1)
            attn_l0_weights = self.dropout(attn_l0_weights)
            out_l0 = torch.matmul(attn_l0_weights, v0)
            # out_l0: (batch, num_heads, seq_len, head_dim)

            # 平均attention weightsを記録
            attn_l0_avg = attn_l0_weights.mean(dim=1)  # (batch, seq_len, l0_len)
        else:
            out_l0 = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=query.device)
            attn_l0_avg = None

        # === L1 Attention ===
        if l1_keys is not None and l1_values is not None:
            l1_len = l1_keys.size(1)

            k1 = l1_keys.view(batch_size, l1_len, self.num_heads, self.head_dim).transpose(1, 2)
            v1 = l1_values.view(batch_size, l1_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores（L1はfull attention、maskなし）
            attn_l1 = torch.matmul(q, k1.transpose(-2, -1)) * self.scale
            attn_l1_weights = F.softmax(attn_l1, dim=-1)
            attn_l1_weights = self.dropout(attn_l1_weights)
            out_l1 = torch.matmul(attn_l1_weights, v1)

            attn_l1_avg = attn_l1_weights.mean(dim=1)
        else:
            out_l1 = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=query.device)
            attn_l1_avg = None

        # === Combine L0 and L1 ===
        if self.combine_mode == "gate":
            # Gate-based mixing
            gate = torch.sigmoid(self.gate)

            # L1がない場合はL0のみ
            if l1_keys is None:
                out = out_l0
            # L0がない場合はL1のみ
            elif l0_keys is None:
                out = out_l1
            else:
                out = gate * out_l0 + (1 - gate) * out_l1

        else:  # concat
            # Concatenation
            out_l0_flat = out_l0.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
            out_l1_flat = out_l1.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
            out = torch.cat([out_l0_flat, out_l1_flat], dim=-1)
            out = self.out_proj(out)

            return DCAOutput(
                output=out,
                l0_attention=attn_l0_avg,
                l1_attention=attn_l1_avg,
            )

        # Reshape and project (gate mode)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        return DCAOutput(
            output=out,
            l0_attention=attn_l0_avg,
            l1_attention=attn_l1_avg,
        )

    def get_gate_value(self) -> float:
        """現在のgate値を取得。"""
        return torch.sigmoid(self.gate).item()


class DualContextLM(nn.Module):
    """
    Dual-Context Attentionを使用した言語モデル層。

    通常のTransformer層にDual-Context Attentionを追加。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        combine_mode: str = "gate",
    ):
        """
        Args:
            dim: 隠れ状態の次元
            num_heads: Attentionヘッド数
            ffn_dim: FFN中間層の次元（Noneの場合は4 * dim）
            dropout: ドロップアウト率
            combine_mode: L0/L1の統合方法
        """
        super().__init__()
        self.dim = dim
        ffn_dim = ffn_dim or dim * 4

        # Dual-Context Attention
        self.dca = DualContextAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            combine_mode=combine_mode,
        )

        # Layer Norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: Tensor,
        l0_keys: Optional[Tensor] = None,
        l0_values: Optional[Tensor] = None,
        l1_keys: Optional[Tensor] = None,
        l1_values: Optional[Tensor] = None,
        l0_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, DCAOutput]:
        """
        Forward pass。

        Args:
            hidden_states: (batch, seq_len, dim)
            l0_keys, l0_values: L0コンテキスト
            l1_keys, l1_values: L1コンテキスト
            l0_mask: L0へのCausal mask

        Returns:
            output: (batch, seq_len, dim)
            dca_output: DCAOutput
        """
        # Pre-norm + DCA
        normed = self.ln1(hidden_states)
        dca_out = self.dca(
            query=normed,
            l0_keys=l0_keys,
            l0_values=l0_values,
            l1_keys=l1_keys,
            l1_values=l1_values,
            l0_mask=l0_mask,
        )
        hidden_states = hidden_states + dca_out.output

        # Pre-norm + FFN
        hidden_states = hidden_states + self.ffn(self.ln2(hidden_states))

        return hidden_states, dca_out
