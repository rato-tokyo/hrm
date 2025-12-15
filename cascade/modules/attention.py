"""
LEGOフレームワーク - Attentionメカニズム

注意: 本フレームワークは事前学習専用です。KVキャッシュは実装していません。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """隠れ次元の半分を回転"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """QueryとKeyにRotary位置埋め込みを適用"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """RoPEとCausalマスク付きMulti-Head Attention（事前学習専用、KVキャッシュなし）"""

    def __init__(self, dim: int, num_heads: int, max_seq_len: int, causal: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.scale = self.head_dim ** -0.5

        # Causalマスクをバッファとして登録
        if causal:
            mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            self.register_buffer('causal_mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Q, K, Vを計算
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPEを適用
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attentionを計算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causalマスクを適用（未来のトークンへのAttentionを防止）
        if self.causal:
            attn = attn.masked_fill(self.causal_mask[:seq_len, :seq_len], float('-inf'))

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)
