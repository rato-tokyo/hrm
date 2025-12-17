"""
DCA-LLM v2: Dual-Context Attention統合言語モデル

L0/L1 2層コンテキストを内蔵したTransformerアーキテクチャ。
- L0: ローカルコンテキスト（ウィンドウ内の詳細なattention）
- L1: 圧縮コンテキスト（ウィンドウ外の要約情報）

GPT-2との公平な比較実験用に設計。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DCALLMOutput:
    """DCA-LLMの出力。"""
    logits: Tensor  # (batch, seq_len, vocab_size)
    loss: Optional[Tensor] = None


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


class IntegratedDCALLM(nn.Module):
    """
    DCAを内部に統合した言語モデル（L0/L1 2層構造）。

    長いシーケンスをウィンドウに分割し、各ウィンドウで:
    - L0: 現在のウィンドウ内でcausal attention
    - L1: 過去のウィンドウを圧縮してattention

    使用例:
        model = IntegratedDCALLM(vocab_size=50257, dim=256, num_layers=4)
        outputs = model(input_ids, labels=labels)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 1024,
        window_size: int = 256,
        compression_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.compression_ratio = compression_ratio

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.embed_dropout = nn.Dropout(dropout)

        # DCA Blocks
        self.blocks = nn.ModuleList([
            IntegratedDCABlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                compression_ratio=compression_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def reset_memory(self):
        """Compatibility method (no-op for training)."""
        pass

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> DCALLMOutput:
        """
        Forward pass with windowed DCA.

        長いシーケンスをwindow_sizeに分割し、各ウィンドウで:
        - L0: 現在のウィンドウ内のcausal attention
        - L1: 過去のウィンドウを圧縮したattention

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) for loss calculation

        Returns:
            DCALLMOutput
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.embed_dropout(hidden_states)

        # Split into windows
        num_windows = (seq_len + self.window_size - 1) // self.window_size
        pad_len = num_windows * self.window_size - seq_len

        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))

        # Process windows
        all_outputs = []

        for window_idx in range(num_windows):
            start_idx = window_idx * self.window_size
            end_idx = start_idx + self.window_size
            window_hidden = hidden_states[:, start_idx:end_idx, :]

            # Causal mask for this window
            causal_mask = torch.tril(torch.ones(
                self.window_size, self.window_size, device=device
            ))

            # Collect past context from previous windows
            if window_idx > 0:
                past_context = hidden_states[:, :start_idx, :]
            else:
                past_context = None

            # Forward through all DCA blocks
            for block in self.blocks:
                window_hidden, _ = block(window_hidden, causal_mask, past_context)

            all_outputs.append(window_hidden)

        # Concatenate all window outputs
        hidden_states = torch.cat(all_outputs, dim=1)

        # Remove padding
        if pad_len > 0:
            hidden_states = hidden_states[:, :seq_len, :]

        # Output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return DCALLMOutput(logits=logits, loss=loss)


def create_integrated_dca_llm(
    vocab_size: int,
    dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    max_seq_len: int = 1024,
    window_size: int = 256,
    compression_ratio: int = 4,
    dropout: float = 0.1,
    device: Optional[str] = None,
) -> IntegratedDCALLM:
    """
    DCA統合言語モデルを作成するファクトリ関数。

    Args:
        vocab_size: 語彙サイズ
        dim: モデル次元
        num_layers: レイヤー数
        num_heads: ヘッド数
        max_seq_len: 最大シーケンス長
        window_size: L0のウィンドウサイズ
        compression_ratio: L1の圧縮率
        dropout: ドロップアウト率
        device: デバイス

    Returns:
        IntegratedDCALLM インスタンス
    """
    model = IntegratedDCALLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        window_size=window_size,
        compression_ratio=compression_ratio,
        dropout=dropout,
    )

    if device:
        model = model.to(device)

    return model
