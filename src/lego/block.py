"""
LEGO Framework - LEGOBlock

TransformerBlock wrapper with early exit capability.
Separates standard transformer functionality from LEGO-specific features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .modules import TransformerBlock


class LEGOBlock(nn.Module):
    """
    TransformerBlock with early exit capability.

    Wraps a standard TransformerBlock and adds:
    - Lightweight exit_classifier for confidence prediction
    - Threshold for token-level early exit decision (set by trainer)

    This separation allows:
    - Standard TransformerBlock to be used independently
    - Easy replacement of transformer implementation (e.g., Flash Attention)
    - Clear distinction between standard and LEGO-specific functionality

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers in this block
        output_head: Shared output projection (reference, not owned)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        output_head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.threshold = 1.0  # Set by trainer
        self.output_head = output_head  # Shared reference, not owned

        # Standard transformer block (composition)
        self.transformer = TransformerBlock(dim, num_heads, num_layers)

        # LEGO-specific: Lightweight exit classifier (dim -> 1)
        self.exit_classifier = nn.Linear(dim, 1)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer with exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            h: Output hidden states (batch_size, seq_len, dim)
            logits: Output logits (batch_size, seq_len, vocab_size)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")

        # Standard transformer forward
        h = self.transformer(h)

        # Output logits
        logits = self.output_head(h)

        # LEGO-specific: Lightweight confidence from exit_classifier
        confidence = torch.sigmoid(self.exit_classifier(h)).squeeze(-1)
        should_exit = confidence >= self.threshold

        return h, logits, should_exit

    def set_output_head(self, output_head: nn.Linear) -> None:
        """Set the shared output head reference."""
        self.output_head = output_head
