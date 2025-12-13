"""
LEGO Framework - LEGOBlock

A block of transformer layers with early exit capability at the final layer.
Each block owns its layers and handles forward pass through them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .modules import TransformerBlock


class LEGOBlock(nn.Module):
    """
    A block of transformer layers with early exit capability.

    Each LEGOBlock:
    - Owns multiple TransformerBlock layers
    - Has a threshold for token-level early exit decision
    - Can compute confidence for routing decisions

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers in this block
        threshold: Confidence threshold for early exit (1.0 = no early exit)
        output_head: Shared output projection (reference, not owned)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        threshold: float = 1.0,
        output_head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.threshold = threshold
        self.output_head = output_head  # Shared reference, not owned

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all layers with exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            h: Output hidden states (batch_size, seq_len, dim)
            logits: Output logits (batch_size, seq_len, vocab_size)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")

        for layer in self.layers:
            h = layer(h)

        logits = self.output_head(h)
        confidence = F.softmax(logits, dim=-1).max(dim=-1).values
        should_exit = confidence >= self.threshold

        return h, logits, should_exit

    def set_output_head(self, output_head: nn.Linear) -> None:
        """Set the shared output head reference."""
        self.output_head = output_head
