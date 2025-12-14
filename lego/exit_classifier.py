"""
LEGO Framework - ExitClassifier

Predicts per-token loss from hidden states using MLP (2-layer).
Low predicted loss = easy token = should exit early.

Architecture decision (2024-12-15):
- Linear Router: 17.2% Oracle
- MLP Router (2-layer): 30.2% Oracle
- MLP provides +13% improvement over Linear.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ExitClassifier(nn.Module):
    """
    MLP-based loss predictor for early exit decision.

    Uses 2-layer MLP to predict per-token loss from hidden states.
    This approach showed 30.2% Oracle performance vs 17.2% for Linear.

    Architecture:
        fc1: Linear(dim, hidden_dim) + ReLU
        fc2: Linear(hidden_dim, 1)

    Exit decision: predicted_loss <= threshold means the token should exit.

    Args:
        dim: Input dimension (model dimension)
        hidden_dim: Hidden layer dimension for MLP
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.threshold = 0.0  # Set by trainer after training

    @property
    def dim(self) -> int:
        """Input dimension."""
        return self.fc1.in_features

    @property
    def hidden_dim(self) -> int:
        """Hidden layer dimension."""
        return self.fc1.out_features

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predicted loss and exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            predicted_loss: Token-level predicted loss (batch_size, seq_len)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        predicted_loss = self._mlp_forward(h)
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
        return self._mlp_forward(h)

    def _mlp_forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        MLP forward pass: fc1 -> ReLU -> fc2.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            predicted_loss: Token-level predicted loss (batch_size, seq_len)
        """
        x = F.relu(self.fc1(h))
        return self.fc2(x).squeeze(-1)
