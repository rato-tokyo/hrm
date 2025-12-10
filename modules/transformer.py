"""
Transformer blocks for HRM
"""

import torch
import torch.nn as nn
from typing import Optional

from .norm import RMSNorm
from .attention import MultiHeadAttention
from .ffn import GatedLinearUnit


class TransformerBlock(nn.Module):
    """Transformer Block with Post-Norm architecture"""

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


class RecurrentModule(nn.Module):
    """
    Recurrent Module (used for both H and L modules)
    Implemented as a stack of Transformer blocks
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
