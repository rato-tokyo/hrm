"""
LEGO Framework - LEGOBlock

TransformerBlock wrapper with CALM-style early exit capability.
Uses cosine similarity between layer input and output for exit decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

from .modules import TransformerBlock
from .exit_classifier import ExitClassifier


class LEGOBlock(nn.Module):
    """
    TransformerBlock with CALM-style early exit capability.

    Wraps a standard TransformerBlock and adds:
    - ExitClassifier (CALM-style cos_sim) for exit decision
    - Shared output_head for logits computation

    CALM-style exit: cos_sim(h_in, h_out) >= threshold means exit.
    High similarity = layer made little change = representation stabilized.

    Args:
        transformer: TransformerBlock to wrap
    """

    def __init__(self, transformer: TransformerBlock):
        super().__init__()
        self.transformer = transformer
        self.exit_classifier = ExitClassifier()
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

    @property
    def threshold(self) -> float:
        """Threshold for early exit (delegated to exit_classifier)."""
        return self.exit_classifier.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set threshold for early exit."""
        self.exit_classifier.threshold = value

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer with CALM-style exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            h_out: Output hidden states (batch_size, seq_len, dim)
            logits: Output logits (batch_size, seq_len, vocab_size)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")

        h_in = h

        # Standard transformer forward
        h_out = self.transformer(h)

        # Output logits
        logits = self.output_head(h_out)

        # CALM-style exit decision: cos_sim(h_in, h_out)
        _, should_exit = self.exit_classifier(h_in, h_out)

        return h_out, logits, should_exit

    def set_output_head(self, output_head: nn.Linear) -> None:
        """Set the shared output head reference."""
        self.output_head = output_head
