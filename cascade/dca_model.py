"""
IntegratedDCALLM: DCAを内部に統合した言語モデル。

L0/L1 2層コンテキストを内蔵したTransformerアーキテクチャ。
GPT-2との公平な比較実験用に設計。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dca_block import IntegratedDCABlock
from .dca_output import DCALLMOutput


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
