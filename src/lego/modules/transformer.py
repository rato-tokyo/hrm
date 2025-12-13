"""
LEGO Framework - Transformer Components

Note: This is a pre-training only framework. KV cache is not implemented.
"""

import torch
import torch.nn as nn
from typing import Optional

from .norm import RMSNorm
from .attention import MultiHeadAttention
from .ffn import GatedLinearUnit


class TransformerLayer(nn.Module):
    """
    Single Transformer Layer (Attention + FFN).

    Post-Norm architecture, pre-training only (no KV cache).
    """

    def __init__(self, dim: int, num_heads: int = 8, ffn_dim: Optional[int] = None):
        super().__init__()
        ffn_dim = ffn_dim or dim * 4

        self.attn = MultiHeadAttention(dim, num_heads)
        self.ffn = GatedLinearUnit(dim, ffn_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-Norm architecture
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerBlock(nn.Module):
    """
    Stack of TransformerLayers.

    Standard transformer block that can be used independently or
    wrapped by LEGOBlock for early exit capability.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers in this block
        ffn_dim: FFN hidden dimension (default: dim * 4)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: Optional[int] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor (batch_size, seq_len, dim)

        Returns:
            Output tensor (batch_size, seq_len, dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x
