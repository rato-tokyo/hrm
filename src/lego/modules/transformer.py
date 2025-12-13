"""
LEGO Framework - Transformer Block
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        # Post-Norm architecture with optional KV cache
        if use_cache:
            attn_out, new_cache = self.attn(x, kv_cache=kv_cache, use_cache=True)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ffn(x))
            return x, new_cache
        else:
            x = self.norm1(x + self.attn(x, kv_cache=kv_cache))
            x = self.norm2(x + self.ffn(x))
            return x
