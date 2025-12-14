"""
LEGO Framework - ExitClassifier

Predicts per-token loss from hidden states (BDR-style approach).
Low predicted loss = easy token = should exit early.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class ExitClassifier(nn.Module):
    """
    Loss predictor for early exit decision.

    Predicts per-token loss directly from hidden states (no sigmoid).
    This is the BDR (Bimodal Distribution Removal) style approach:
    - Train to predict actual cross-entropy loss
    - Low predicted loss → easy token → should exit
    - High predicted loss → hard token → continue to next block

    Exit decision: predicted_loss <= threshold means the token should exit.

    Args:
        dim: Input dimension (model dimension)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.threshold = 0.0  # Set by trainer after training

    @property
    def dim(self) -> int:
        """Input dimension."""
        return self.linear.in_features

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predicted loss and exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            predicted_loss: Token-level predicted loss (batch_size, seq_len)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        predicted_loss = self.linear(h).squeeze(-1)
        should_exit = predicted_loss <= self.threshold
        return predicted_loss, should_exit

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted loss (kept as compute_confidence for API compatibility).

        Note: Lower value = higher confidence = easier token.
        This is the opposite of the old sigmoid-based approach.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            predicted_loss: Token-level predicted loss (batch_size, seq_len)
        """
        return self.linear(h).squeeze(-1)
