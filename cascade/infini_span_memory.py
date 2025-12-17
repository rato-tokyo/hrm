"""
CASCADEフレームワーク - Infini-Span Memory

Infini-Attention風のlocal/global分離メモリ。
- Local Memory: 最新2 span分のトークンを保持
- Global Memory: 過去spanの圧縮ベクトルを保持

span切り替わり時に双方向エンコーダで圧縮してglobalに格納。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SpanInfo:
    """Span情報を保持するデータクラス。"""
    start_pos: int  # 元シーケンスでの開始位置
    end_pos: int    # 元シーケンスでの終了位置
    hidden_states: Tensor  # (seq_len, dim)

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos + 1


@dataclass
class MemoryState:
    """メモリの状態を表すデータクラス。"""
    # Local Memory: 最新2 span
    local_spans: List[SpanInfo] = field(default_factory=list)

    # Global Memory: 圧縮されたspan表現
    global_vectors: List[Tensor] = field(default_factory=list)  # List of (dim,)
    global_positions: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) pairs

    # 処理済みの総トークン数
    total_processed: int = 0


class BidirectionalSpanEncoder(nn.Module):
    """
    双方向エンコーダでspanを圧縮。

    Spanの隠れ状態を受け取り、平均プーリングで1ベクトルに圧縮。
    オプションで学習可能な変換を適用。
    """

    def __init__(
        self,
        dim: int,
        use_projection: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: 隠れ状態の次元
            use_projection: 圧縮後に線形変換を適用するか
            dropout: ドロップアウト率
        """
        super().__init__()
        self.dim = dim
        self.use_projection = use_projection

        if use_projection:
            self.projection: nn.Module = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Spanを1ベクトルに圧縮。

        Args:
            hidden_states: (seq_len, dim) or (batch, seq_len, dim)

        Returns:
            compressed: (dim,) or (batch, dim)
        """
        # 平均プーリング
        if hidden_states.dim() == 2:
            # (seq_len, dim) -> (dim,)
            pooled = hidden_states.mean(dim=0)
        else:
            # (batch, seq_len, dim) -> (batch, dim)
            pooled = hidden_states.mean(dim=1)

        # 射影（オプション）
        compressed = self.projection(pooled)

        return compressed


class InfiniSpanMemory(nn.Module):
    """
    Infini-Attention風のLocal/Global分離メモリ。

    アーキテクチャ:
    - Local Memory: 最新2 span分のトークン隠れ状態を保持
    - Global Memory: 過去spanの圧縮ベクトルを保持

    使用例:
        memory = InfiniSpanMemory(dim=768, span_size=128)

        # ストリーミング処理
        for hidden_chunk in stream_hidden_states:
            local_ctx, global_ctx = memory.update(hidden_chunk)
            # local_ctx: 最新2 span分の隠れ状態
            # global_ctx: 過去spanの圧縮ベクトル群
    """

    def __init__(
        self,
        dim: int,
        span_size: int = 128,
        num_local_spans: int = 2,
        max_global_spans: int = 256,
        use_projection: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: 隠れ状態の次元
            span_size: 固定span長
            num_local_spans: ローカルに保持するspan数（デフォルト: 2）
            max_global_spans: グローバルに保持する最大span数
            use_projection: 圧縮時に射影を使用するか
            dropout: ドロップアウト率
        """
        super().__init__()
        self.dim = dim
        self.span_size = span_size
        self.num_local_spans = num_local_spans
        self.max_global_spans = max_global_spans

        # 双方向エンコーダ（span圧縮用）
        self.span_encoder = BidirectionalSpanEncoder(
            dim=dim,
            use_projection=use_projection,
            dropout=dropout,
        )

        # メモリ状態
        self._state = MemoryState()

    def reset(self):
        """メモリをリセット。"""
        self._state = MemoryState()

    @property
    def state(self) -> MemoryState:
        """現在のメモリ状態。"""
        return self._state

    def update(
        self,
        hidden_states: Tensor,
        force_span_boundary: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        新しい隠れ状態でメモリを更新。

        Args:
            hidden_states: (seq_len, dim) - 新しいトークンの隠れ状態
            force_span_boundary: 強制的にspan境界とするか

        Returns:
            local_context: (local_len, dim) - ローカルコンテキスト
            global_context: (num_global, dim) or None - グローバルコンテキスト
        """
        seq_len = hidden_states.size(0)

        # 現在のlocalに追加
        current_pos = self._state.total_processed

        # 新しいSpanInfoを作成
        new_span = SpanInfo(
            start_pos=current_pos,
            end_pos=current_pos + seq_len - 1,
            hidden_states=hidden_states.detach(),
        )

        # 累積hidden statesの管理
        if self._state.local_spans:
            # 既存のlocalと新しいhidden statesを結合
            last_span = self._state.local_spans[-1]
            combined_hidden = torch.cat([
                last_span.hidden_states,
                hidden_states
            ], dim=0)

            # span境界を判定（固定サイズ超過 or 強制）
            if combined_hidden.size(0) >= self.span_size or force_span_boundary:
                # spanを分割
                self._split_and_update_spans(combined_hidden, last_span.start_pos)
            else:
                # 結合を継続
                self._state.local_spans[-1] = SpanInfo(
                    start_pos=last_span.start_pos,
                    end_pos=current_pos + seq_len - 1,
                    hidden_states=combined_hidden.detach(),
                )
        else:
            # 最初のspan
            self._state.local_spans.append(new_span)

        self._state.total_processed += seq_len

        return self._get_contexts()

    def _split_and_update_spans(
        self,
        hidden_states: Tensor,
        start_pos: int,
    ):
        """
        hidden statesをspan単位に分割してメモリを更新。
        """
        total_len = hidden_states.size(0)
        current_pos = start_pos
        new_local_spans: List[SpanInfo] = []

        idx = 0
        while idx < total_len:
            end_idx = min(idx + self.span_size, total_len)
            span_hidden = hidden_states[idx:end_idx]

            span_info = SpanInfo(
                start_pos=current_pos,
                end_pos=current_pos + (end_idx - idx) - 1,
                hidden_states=span_hidden.detach(),
            )
            new_local_spans.append(span_info)

            current_pos += (end_idx - idx)
            idx = end_idx

        # localスパンを更新（最新num_local_spans個を保持）
        all_spans = new_local_spans

        if len(all_spans) > self.num_local_spans:
            # 古いspanをglobalに移動
            spans_to_compress = all_spans[:-self.num_local_spans]
            self._state.local_spans = all_spans[-self.num_local_spans:]

            for span in spans_to_compress:
                self._compress_to_global(span)
        else:
            self._state.local_spans = all_spans

    def _compress_to_global(self, span: SpanInfo):
        """
        Spanを圧縮してglobalメモリに追加。
        """
        # 双方向エンコーダで圧縮
        compressed = self.span_encoder(span.hidden_states)

        self._state.global_vectors.append(compressed.detach())
        self._state.global_positions.append((span.start_pos, span.end_pos))

        # 最大数を超えたら古いものを削除
        if len(self._state.global_vectors) > self.max_global_spans:
            self._state.global_vectors.pop(0)
            self._state.global_positions.pop(0)

    def _get_contexts(self) -> Tuple[Tensor, Optional[Tensor]]:
        """
        現在のlocal/globalコンテキストを取得。
        """
        # Local context: 全localスパンを結合
        if self._state.local_spans:
            local_hidden = torch.cat([
                span.hidden_states for span in self._state.local_spans
            ], dim=0)
        else:
            local_hidden = torch.zeros(0, self.dim)

        # Global context: 圧縮ベクトルをスタック
        if self._state.global_vectors:
            global_hidden = torch.stack(self._state.global_vectors, dim=0)
        else:
            global_hidden = None

        return local_hidden, global_hidden

    def get_full_context(self) -> Tensor:
        """
        Local + Global を結合した完全なコンテキストを取得。

        Returns:
            context: (local_len + num_global, dim)
        """
        local_ctx, global_ctx = self._get_contexts()

        if global_ctx is not None:
            # Globalを先頭に配置
            return torch.cat([global_ctx, local_ctx], dim=0)
        else:
            return local_ctx

    def get_attention_mask(self) -> Tuple[Tensor, Tensor]:
        """
        Local Attention用とGlobal Attention用のマスクを取得。

        Returns:
            local_mask: (local_len, local_len) - Causal mask for local
            global_mask: (local_len, num_global) - Full attention to global
        """
        local_ctx, global_ctx = self._get_contexts()
        local_len = local_ctx.size(0)
        device = local_ctx.device

        # Local: Causal mask（下三角）
        local_mask = torch.tril(torch.ones(local_len, local_len, device=device))

        # Global: 全位置からglobalへのattentionを許可
        if global_ctx is not None:
            num_global = global_ctx.size(0)
            global_mask = torch.ones(local_len, num_global, device=device)
        else:
            global_mask = torch.zeros(local_len, 0, device=device)

        return local_mask, global_mask


class InfiniSpanAttention(nn.Module):
    """
    Local AttentionとGlobal Attentionを統合するモジュール。

    Local: 最新2 span内でのCausal Attention
    Global: 過去span圧縮ベクトルへのCross Attention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init: float = 0.0,
    ):
        """
        Args:
            dim: 隠れ状態の次元
            num_heads: Attentionヘッド数
            dropout: ドロップアウト率
            gate_init: Local/Global混合ゲートの初期値
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Global用のKey, Value projections（別パラメータ）
        self.k_global_proj = nn.Linear(dim, dim)
        self.v_global_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Local/Global混合ゲート（学習可能）
        self.gate = nn.Parameter(torch.tensor(gate_init))

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        local_hidden: Tensor,
        global_hidden: Optional[Tensor] = None,
        local_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Local + Global Attentionを実行。

        Args:
            local_hidden: (batch, local_len, dim)
            global_hidden: (batch, num_global, dim) or None
            local_mask: (local_len, local_len) - Causal mask

        Returns:
            output: (batch, local_len, dim)
        """
        batch_size, local_len, _ = local_hidden.shape

        # Query (from local)
        q = self.q_proj(local_hidden)
        q = q.view(batch_size, local_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === Local Attention ===
        k_local = self.k_proj(local_hidden)
        v_local = self.v_proj(local_hidden)
        k_local = k_local.view(batch_size, local_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_local = v_local.view(batch_size, local_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores (local)
        attn_local = torch.matmul(q, k_local.transpose(-2, -1)) * self.scale

        # Causal mask適用
        if local_mask is not None:
            attn_local = attn_local.masked_fill(local_mask == 0, float('-inf'))

        attn_local = torch.softmax(attn_local, dim=-1)
        attn_local = self.dropout(attn_local)
        out_local = torch.matmul(attn_local, v_local)

        # === Global Attention ===
        if global_hidden is not None and global_hidden.size(1) > 0:
            num_global = global_hidden.size(1)

            k_global = self.k_global_proj(global_hidden)
            v_global = self.v_global_proj(global_hidden)
            k_global = k_global.view(batch_size, num_global, self.num_heads, self.head_dim).transpose(1, 2)
            v_global = v_global.view(batch_size, num_global, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores (global) - no mask, full attention
            attn_global = torch.matmul(q, k_global.transpose(-2, -1)) * self.scale
            attn_global = torch.softmax(attn_global, dim=-1)
            attn_global = self.dropout(attn_global)
            out_global = torch.matmul(attn_global, v_global)

            # Gate-based mixing
            gate = torch.sigmoid(self.gate)
            out = gate * out_local + (1 - gate) * out_global
        else:
            out = out_local

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, local_len, self.dim)
        out = self.out_proj(out)

        return out
