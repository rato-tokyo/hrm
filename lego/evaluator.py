"""
LEGO Framework - Model Evaluation

Functions for evaluating LEGOLLM with TRUE Early Exit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import LEGOLLM


def evaluate_legollm(
    model: "LEGOLLM",
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Evaluate LEGOLLM with TRUE Early Exit.

    Args:
        model: Trained LEGOLLM model
        val_batches: List of (x, y) batches for validation

    Returns:
        Dict with ppl, accuracy, shallow_ratio, compute_cost, exit_counts
    """
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct = 0
    aggregated_exit_counts: List[int] = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            logits, stats = model.forward(x, return_stats=True)

            # Accumulate exit counts
            batch_exit_counts: List[int] = stats['exit_counts']
            if not aggregated_exit_counts:
                aggregated_exit_counts = [0] * len(batch_exit_counts)
            for i, count in enumerate(batch_exit_counts):
                aggregated_exit_counts[i] += count

            # Loss and accuracy
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            total_loss += F.cross_entropy(logits_flat, y_flat, reduction='sum').item()
            total_tokens += y_flat.numel()
            correct += int((logits_flat.argmax(dim=-1) == y_flat).sum().item())

    ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
    acc = correct / total_tokens if total_tokens > 0 else 0.0

    # Compute statistics using model's method
    exit_stats = model._compute_exit_stats(aggregated_exit_counts)

    return {
        'ppl': ppl,
        'accuracy': acc,
        'shallow_ratio': exit_stats['shallow_ratio'],
        'compute_cost': exit_stats['compute_cost'],
        'compute_savings': 1.0 - exit_stats['compute_cost'],
        'exit_counts': aggregated_exit_counts,
        'total_tokens': total_tokens,
    }
