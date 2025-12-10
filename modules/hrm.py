"""
Hierarchical Reasoning Model (HRM) main module
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

from .transformer import RecurrentModule


class LowLevelModule(nn.Module):
    """
    Low-Level Module (L-module)
    Fast, detailed computations
    Takes: previous L state, current H state, input embedding
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.recurrent = RecurrentModule(dim, num_layers, num_heads)

    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        combined = z_L + z_H + x_embed
        return self.recurrent(combined)


class HighLevelModule(nn.Module):
    """
    High-Level Module (H-module)
    Slow, abstract planning
    Takes: previous H state, current L state
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.recurrent = RecurrentModule(dim, num_layers, num_heads)

    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        combined = z_H + z_L
        return self.recurrent(combined)


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model

    Args:
        vocab_size: Size of vocabulary for input/output tokens
        dim: Hidden dimension
        num_layers: Number of Transformer layers per module
        num_heads: Number of attention heads
        seq_len: Sequence length
        N: Number of high-level cycles
        T: Number of low-level timesteps per cycle
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        seq_len: int = 81,
        N: int = 2,
        T: int = 4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.N = N
        self.T = T

        # Input embedding
        self.input_embedding = nn.Embedding(vocab_size, dim)

        # Initial hidden states (learnable)
        self.z_L_init = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.z_H_init = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        # Recurrent modules
        self.L_module = LowLevelModule(dim, num_layers, num_heads)
        self.H_module = HighLevelModule(dim, num_layers, num_heads)

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Q-head for ACT
        self.q_head = nn.Linear(dim, 2, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using truncated LeCun Normal"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial hidden states for H and L modules"""
        z_H = self.z_H_init.expand(batch_size, -1, -1)
        z_L = self.z_L_init.expand(batch_size, -1, -1)
        return z_H, z_L

    def forward_pass(
        self,
        x: torch.Tensor,
        z_H: Optional[torch.Tensor] = None,
        z_L: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass of HRM (N cycles of T timesteps each)
        """
        batch_size = x.shape[0]
        x_embed = self.input_embedding(x)

        if z_H is None or z_L is None:
            z_H, z_L = self.get_initial_state(batch_size)

        # 1-step gradient approximation
        with torch.no_grad():
            for i in range(self.N * self.T - 1):
                z_L = self.L_module(z_L, z_H, x_embed)
                if (i + 1) % self.T == 0:
                    z_H = self.H_module(z_H, z_L)

        # Final step with gradient
        z_L = self.L_module(z_L, z_H, x_embed)
        z_H = self.H_module(z_H, z_L)

        y_hat = self.output_head(z_H)
        q_values = torch.sigmoid(self.q_head(z_H.mean(dim=1)))

        return z_H, z_L, y_hat, q_values

    def forward(
        self,
        x: torch.Tensor,
        num_segments: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with deep supervision"""
        z_H, z_L = None, None

        for _ in range(num_segments):
            z_H, z_L, y_hat, q_values = self.forward_pass(x, z_H, z_L)
            z_H = z_H.detach()
            z_L = z_L.detach()

        return y_hat, q_values


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
