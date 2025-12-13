"""
LEGO Framework - LEGOBlock

A block of transformer layers with early exit capability at the final layer.
Each block owns its layers and handles forward pass through them.
"""

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
    - Can be trained independently (for hard example training)

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

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers in this block.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            Output hidden states (batch_size, seq_len, dim)
        """
        for layer in self.layers:
            h = layer(h)
        return h

    def set_output_head(self, output_head: nn.Linear) -> None:
        """Set the shared output head reference."""
        self.output_head = output_head

    def compute_confidence(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute confidence (max probability) from hidden state.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            confidence: Max probability per token (batch_size, seq_len)
        """
        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return logits, confidence

    def forward_with_exit(
        self,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with exit decision.

        Combines forward(), compute_confidence(), and exit mask computation
        into a single method. This encapsulates the block's exit logic.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            h: Output hidden states (batch_size, seq_len, dim)
            logits: Output logits (batch_size, seq_len, vocab_size)
            exit_mask: Boolean mask where True = should exit (batch_size, seq_len)
        """
        h = self.forward(h)
        logits, confidence = self.compute_confidence(h)
        exit_mask = confidence >= self.threshold
        return h, logits, exit_mask

    def freeze(self) -> None:
        """Freeze all parameters in this block."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters in this block."""
        for param in self.parameters():
            param.requires_grad = True
