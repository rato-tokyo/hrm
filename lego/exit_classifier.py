"""
LEGO Framework - ExitClassifier

Confidence computation and early exit decision for LEGOBlock.
Uses loss-based labels: exp(-cross_entropy_loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class ExitClassifier(nn.Module):
    """
    Confidence computation and early exit decision.

    Computes token-level confidence using a lightweight Linear layer.
    Exit decision: confidence >= threshold means the token should exit.

    Trained with loss-based labels: exp(-cross_entropy_loss).
    This approach showed best results in experiments.

    Args:
        dim: Input dimension (model dimension)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.threshold = 1.0  # Set by trainer after training

    @property
    def dim(self) -> int:
        """Input dimension."""
        return self.linear.in_features

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute confidence and exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            confidence: Token-level confidence (batch_size, seq_len)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        confidence = torch.sigmoid(self.linear(h)).squeeze(-1)
        should_exit = confidence >= self.threshold
        return confidence, should_exit

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence only (without exit decision).

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            confidence: Token-level confidence (batch_size, seq_len)
        """
        return torch.sigmoid(self.linear(h)).squeeze(-1)
