"""
LEGO Framework - Transformer Components

Note: This is a pre-training only framework. KV cache is not implemented.
"""

import torch
import torch.nn as nn

from .norm import RMSNorm
from .attention import MultiHeadAttention
from .ffn import GatedLinearUnit


class TransformerLayer(nn.Module):
    """
    Single Transformer Layer (Attention + FFN).

    Post-Norm architecture, pre-training only (no KV cache).
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, max_seq_len: int, causal: bool, eps: float):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, max_seq_len, causal)
        self.ffn = GatedLinearUnit(dim, ffn_dim)
        self.norm1 = RMSNorm(dim, eps)
        self.norm2 = RMSNorm(dim, eps)

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
        ffn_dim: FFN hidden dimension
        max_seq_len: Maximum sequence length
        causal: Whether to use causal masking
        eps: Epsilon for RMSNorm
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
