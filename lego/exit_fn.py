"""
LEGO Framework - Exit Functions

Functions for determining early exit based on hidden_history.
Default implementation uses CALM-style cosine similarity.
"""

import torch
import torch.nn.functional as F
from typing import List, Callable


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


def compute_cos_sim(h_in: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two hidden states.

    Args:
        h_in: Input hidden states (batch_size, seq_len, dim)
        h_out: Output hidden states (batch_size, seq_len, dim)

    Returns:
        cos_sim: Cosine similarity per token (batch_size, seq_len)
    """
    h_in_norm = F.normalize(h_in, dim=-1)
    h_out_norm = F.normalize(h_out, dim=-1)
    return (h_in_norm * h_out_norm).sum(dim=-1)
