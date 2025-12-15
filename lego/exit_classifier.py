"""
LEGO Framework - ExitClassifier (CALM Style)

Uses cosine similarity between layer input and output for exit decision.
No training required - purely based on representation stability.

CALM paper reference:
"State Propagation: the cosine similarity between the hidden states
of consecutive layers"

High cos_sim = small change in layer = representation stabilized = should exit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ExitClassifier(nn.Module):
    """
    CALM-style exit classifier using cosine similarity.

    Computes cos_sim(h_in, h_out) to determine if representation has stabilized.
    High similarity means the layer made little change, suggesting saturation.

    No parameters to train - threshold is set based on quantile analysis.
    """

    def __init__(self):
        super().__init__()
        self.threshold = 0.0  # Set by trainer after analysis

    def forward(
        self, h_in: torch.Tensor, h_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine similarity and exit decision.

        Args:
            h_in: Input hidden states (batch_size, seq_len, dim)
            h_out: Output hidden states (batch_size, seq_len, dim)

        Returns:
            cos_sim: Cosine similarity per token (batch_size, seq_len)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        cos_sim = self.compute_similarity(h_in, h_out)
        should_exit = cos_sim >= self.threshold
        return cos_sim, should_exit

    def compute_similarity(
        self, h_in: torch.Tensor, h_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between h_in and h_out.

        Args:
            h_in: Input hidden states (batch_size, seq_len, dim)
            h_out: Output hidden states (batch_size, seq_len, dim)

        Returns:
            cos_sim: Cosine similarity per token (batch_size, seq_len)
        """
        h_in_norm = F.normalize(h_in, dim=-1)
        h_out_norm = F.normalize(h_out, dim=-1)
        return (h_in_norm * h_out_norm).sum(dim=-1)
