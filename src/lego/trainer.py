"""
LEGO Framework - Trainer

Training and evaluation for LEGO models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class Trainer:
    """
    Trainer for LEGO models.

    Args:
        vocab_size: Vocabulary size
        device: Device to run on ('cpu' or 'cuda')
    """

    def __init__(self, vocab_size: int, device: str = 'cpu'):
        self.vocab_size = vocab_size
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        use_routing: bool = False
    ) -> Dict[str, float]:
        """Evaluate model with optional early exit routing.

        Args:
            model: Model to evaluate
            val_batches: Validation data batches
            use_routing: If True, use model's block thresholds for routing
        """
        model.eval()

        total_loss = 0.0
        total_correct: float = 0
        total_tokens = 0
        total_shallow = 0.0
        total_compute = 0.0

        for x, y in val_batches:
            x, y = x.to(self.device), y.to(self.device)

            if use_routing:
                logits, stats = model(x, return_stats=True)
                total_shallow += stats['shallow_ratio'] * x.numel()
                total_compute += stats['compute_cost']
            else:
                logits = model(x)

            loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        total_all_tokens = sum(x.shape[0] * x.shape[1] for x, _ in val_batches)

        return {
            'ppl': float(np.exp(avg_loss)),
            'acc': total_correct / total_tokens,
            'shallow_ratio': total_shallow / total_all_tokens if use_routing else 0.0,
            'compute_cost': total_compute / len(val_batches) if use_routing else 1.0,
        }


