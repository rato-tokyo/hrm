"""
Layer-wise Progressive Training (LPT) Transformer

A Transformer that computes loss at each layer's output using a shared output head.
Unlike traditional Deep Supervision which adds auxiliary heads at each layer,
LPT uses the same output head for all layers.

Training approach:
- Each layer's output is passed through the shared output head
- Loss is computed at each layer
- All losses are summed and backpropagated together

This tests whether "supervising intermediate representations" helps,
without adding extra parameters for auxiliary heads.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, List

from .transformer import TransformerBlock


class LayerProgressiveTransformer(nn.Module):
    """
    Transformer with Layer-wise Progressive Training (LPT).

    Each layer's output goes through the shared output head for loss computation.
    No auxiliary heads - uses the same output projection for all layers.

    Training modes:
    1. 'sum': Sum all layer losses, single backward pass
    2. 'progressive': Train layer by layer (freeze previous layers)
    3. 'weighted': Weighted sum of layer losses (deeper layers weighted more)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        seq_len: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

        # Shared output head (used for all layers)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning outputs from all layers.

        Args:
            x: Input tokens [batch, seq_len]

        Returns:
            List of outputs [y_hat_1, y_hat_2, ..., y_hat_L]
            Each y_hat_i is [batch, seq_len, vocab_size]
        """
        h = self.embedding(x)
        outputs = []

        for layer in self.layers:
            h = layer(h)
            # Use shared output head for this layer's output
            y_hat = self.output_head(h)
            outputs.append(y_hat)

        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (only final layer output).

        Args:
            x: Input tokens [batch, seq_len]

        Returns:
            y_hat: Final output [batch, seq_len, vocab_size]
        """
        h = self.embedding(x)

        for layer in self.layers:
            h = layer(h)

        return self.output_head(h)


def compute_lpt_loss(
    outputs: List[torch.Tensor],
    target: torch.Tensor,
    vocab_size: int,
    mode: str = 'sum',
    layer_weights: List[float] = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute Layer-wise Progressive Training loss.

    Args:
        outputs: List of layer outputs [y_hat_1, ..., y_hat_L]
        target: Target tokens [batch, seq_len]
        vocab_size: Vocabulary size
        mode: 'sum', 'weighted', or 'final_only'
        layer_weights: Weights for each layer (for 'weighted' mode)

    Returns:
        total_loss: Combined loss for backpropagation
        layer_losses: Individual loss for each layer (for logging)
    """
    num_layers = len(outputs)
    layer_losses = []

    for i, y_hat in enumerate(outputs):
        loss = nn.functional.cross_entropy(
            y_hat.view(-1, vocab_size), target.view(-1)
        )
        layer_losses.append(loss)

    if mode == 'sum':
        # Equal weight for all layers
        total_loss = sum(layer_losses)
    elif mode == 'weighted':
        # Weighted sum (default: deeper layers have higher weight)
        if layer_weights is None:
            # Linear weighting: [1, 2, 3, 4] for 4 layers
            layer_weights = [i + 1 for i in range(num_layers)]
        total_loss = sum(w * l for w, l in zip(layer_weights, layer_losses))
    elif mode == 'final_only':
        # Only use final layer (for comparison)
        total_loss = layer_losses[-1]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return total_loss, layer_losses
