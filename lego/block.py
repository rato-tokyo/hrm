"""
LEGO Framework - LEGOBlock

TransformerBlock wrapper with early exit capability.
Exit decision is made by exit_fn using hidden_history from all layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Callable, Optional

from .modules import TransformerBlock


# Type alias for exit function
ExitFn = Callable[[List[torch.Tensor], float], torch.Tensor]


def default_exit_fn(hidden_history: List[torch.Tensor], threshold: float) -> torch.Tensor:
    """
    Default CALM-style exit function using cosine similarity.

    Compares the last two hidden states (input and output of final layer).
    High similarity means small change, suggesting saturation.

    Args:
        hidden_history: List of hidden states [input, layer1_out, ...]
        threshold: Cosine similarity threshold for exit

    Returns:
        should_exit: Boolean mask (batch_size, seq_len) where True = should exit
    """
    h_in = hidden_history[-2]   # Input to last layer
    h_out = hidden_history[-1]  # Output of last layer

    # Cosine similarity
    h_in_norm = F.normalize(h_in, dim=-1)
    h_out_norm = F.normalize(h_out, dim=-1)
    cos_sim = (h_in_norm * h_out_norm).sum(dim=-1)

    return cos_sim >= threshold


class LEGOBlock(nn.Module):
    """
    TransformerBlock with early exit capability.

    Wraps a standard TransformerBlock and adds:
    - Exit decision via exit_fn (default: CALM-style cos_sim)
    - Shared output_head for logits computation
    - Access to hidden_history for flexible exit criteria

    Args:
        transformer: TransformerBlock to wrap
        exit_fn: Optional custom exit function. If None, uses default_exit_fn (CALM-style)
    """

    def __init__(
        self,
        transformer: TransformerBlock,
        exit_fn: Optional[ExitFn] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.exit_fn = exit_fn or default_exit_fn
        self.threshold = 0.0  # Set by trainer
        self.output_head: nn.Linear | None = None  # Set by LEGOLLM

    @property
    def dim(self) -> int:
        """Model dimension (delegated to transformer)."""
        return self.transformer.dim

    @property
    def num_heads(self) -> int:
        """Number of attention heads (delegated to transformer)."""
        return self.transformer.num_heads

    @property
    def num_layers(self) -> int:
        """Number of layers (delegated to transformer)."""
        return self.transformer.num_layers

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through transformer with exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            h_out: Output hidden states (batch_size, seq_len, dim)
            logits: Output logits (batch_size, seq_len, vocab_size)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
            hidden_history: List of all hidden states for analysis
        """
        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")

        # Transformer forward with hidden history
        h_out, hidden_history = self.transformer(h)

        # Output logits
        logits = self.output_head(h_out)

        # Exit decision using exit_fn
        should_exit = self.exit_fn(hidden_history, self.threshold)

        return h_out, logits, should_exit, hidden_history

    def set_output_head(self, output_head: nn.Linear) -> None:
        """Set the shared output head reference."""
        self.output_head = output_head
