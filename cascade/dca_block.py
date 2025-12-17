"""
IntegratedDCABlock: DCAを内蔵したTransformerブロック。

L0/L1 2層コンテキスト構造:
- L0: ローカルコンテキスト（ウィンドウ内の詳細なattention）
- L1: 圧縮コンテキスト（ウィンドウ外の要約情報）
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IntegratedDCABlock(nn.Module):
    """
    DCAを内蔵したTransformerブロック（L0/L1 2層構造）。

    L0: ローカルコンテキスト（ウィンドウ内の詳細なattention）
    L1: 圧縮コンテキスト（ウィンドウ外の要約情報）

    訓練時は長いシーケンスを分割してL0/L1を使用。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 256,
        compression_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.compression_ratio = compression_ratio

        # Q projection (shared)
        self.q_proj = nn.Linear(dim, dim)

        # L0: Local context (within window)
        self.l0_k_proj = nn.Linear(dim, dim)
        self.l0_v_proj = nn.Linear(dim, dim)

        # L1: Compressed context (outside window)
        self.l1_k_proj = nn.Linear(dim, dim)
        self.l1_v_proj = nn.Linear(dim, dim)

        # Compression layer for L1 (average pooling + linear projection)
        self.l1_compressor = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Gating mechanism to balance L0 and L1
        self.gate = nn.Linear(dim, 2)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def _compress_context(self, hidden: Tensor) -> Tensor:
        """
        過去のコンテキストを圧縮してL1表現を作成。

        Args:
            hidden: (batch, past_len, dim)

        Returns:
            compressed: (batch, past_len // compression_ratio, dim)
        """
        batch_size, seq_len, dim = hidden.shape

        if seq_len == 0:
            return hidden

        # Pad to make divisible by compression_ratio
        pad_len = (self.compression_ratio - seq_len % self.compression_ratio) % self.compression_ratio
        if pad_len > 0:
            hidden = F.pad(hidden, (0, 0, 0, pad_len))

        # Reshape and average pool
        new_len = hidden.size(1) // self.compression_ratio
        hidden = hidden.view(batch_size, new_len, self.compression_ratio, dim)
        compressed = hidden.mean(dim=2)  # (batch, new_len, dim)

        # Project
        compressed = self.l1_compressor(compressed)
        return compressed

    def forward(
        self,
        hidden_states: Tensor,
        causal_mask: Tensor,
        past_context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with dual-context attention.

        Args:
            hidden_states: (batch, seq_len, dim) - 現在のウィンドウ
            causal_mask: (seq_len, seq_len) - causal mask for L0
            past_context: (batch, past_len, dim) - 過去のコンテキスト（L1用）

        Returns:
            output: (batch, seq_len, dim)
            current_context: hidden_states（次のブロック用）
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pre-norm
        normed = self.ln1(hidden_states)

        # Query projection
        q = self.q_proj(normed)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === L0: Local Context Attention ===
        l0_k = self.l0_k_proj(normed)
        l0_v = self.l0_v_proj(normed)
        l0_k = l0_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        l0_v = l0_v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # L0 attention with causal mask
        attn_l0 = torch.matmul(q, l0_k.transpose(-2, -1)) * self.scale
        attn_l0 = attn_l0.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn_l0 = F.softmax(attn_l0, dim=-1)
        attn_l0 = self.dropout(attn_l0)
        out_l0 = torch.matmul(attn_l0, l0_v)  # (batch, heads, seq, head_dim)

        # === L1: Compressed Context Attention ===
        if past_context is not None and past_context.size(1) > 0:
            # Compress past context
            compressed = self._compress_context(past_context)
            comp_len = compressed.size(1)

            # L1 K/V projections
            l1_k = self.l1_k_proj(compressed)
            l1_v = self.l1_v_proj(compressed)
            l1_k = l1_k.view(batch_size, comp_len, self.num_heads, self.head_dim).transpose(1, 2)
            l1_v = l1_v.view(batch_size, comp_len, self.num_heads, self.head_dim).transpose(1, 2)

            # L1 attention (no causal mask - all past is visible)
            attn_l1 = torch.matmul(q, l1_k.transpose(-2, -1)) * self.scale
            attn_l1 = F.softmax(attn_l1, dim=-1)
            attn_l1 = self.dropout(attn_l1)
            out_l1 = torch.matmul(attn_l1, l1_v)  # (batch, heads, seq, head_dim)

            # Gating: learn to balance L0 and L1
            gate_input = normed.mean(dim=1)  # (batch, dim)
            gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # (batch, 2)
            gate_l0 = gate_weights[:, 0].view(batch_size, 1, 1, 1)
            gate_l1 = gate_weights[:, 1].view(batch_size, 1, 1, 1)

            # Combine L0 and L1
            out = gate_l0 * out_l0 + gate_l1 * out_l1
        else:
            # No past context, use L0 only
            out = out_l0

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        # Residual connection
        hidden_states = hidden_states + out

        # FFN with pre-norm and residual
        hidden_states = hidden_states + self.ffn(self.ln2(hidden_states))

        return hidden_states, normed  # Return normed as context for next window
