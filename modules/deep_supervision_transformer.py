"""
Deep Supervision Transformer

A standard Transformer-based language model that uses deep supervision training
(computing loss at each segment, like HRM) but without the hierarchical structure.

This allows us to compare:
- Standard Transformer: Normal forward pass, single loss
- Deep Supervision Transformer: Multiple forward passes with intermediate losses
- HRM: Hierarchical structure with deep supervision
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

from .transformer import TransformerBlock


class DeepSupervisionTransformer(nn.Module):
    """
    Transformer language model with deep supervision training support.

    Unlike HRM which has hierarchical layers with different update frequencies,
    this is a standard Transformer that simply supports the same training paradigm:
    - Multiple forward passes (segments)
    - Loss computed at each segment
    - States carried across segments

    The "state" in this case is the hidden representation after the Transformer layers,
    which gets fed back as a residual in the next segment.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        seq_len: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Learnable initial state (like HRM)
        self.init_state = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using truncated LeCun Normal"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """Get initial hidden state"""
        return self.init_state.expand(batch_size, -1, -1)

    def forward_pass(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass with optional previous state.

        Args:
            x: Input tokens [batch, seq_len]
            prev_state: Previous hidden state [batch, seq_len, dim] or None

        Returns:
            state: Current hidden state [batch, seq_len, dim]
            y_hat: Output predictions [batch, seq_len, vocab_size]
        """
        batch_size = x.shape[0]

        # Get input embedding
        x_embed = self.embedding(x)

        # If no previous state, use learnable initial state
        if prev_state is None:
            prev_state = self.get_initial_state(batch_size)

        # Combine input with previous state (residual connection)
        h = x_embed + prev_state

        # Pass through Transformer layers
        for layer in self.layers:
            h = layer(h)

        # Output prediction
        y_hat = self.output_head(h)

        return h, y_hat

    def forward(
        self,
        x: torch.Tensor,
        num_segments: int = 1
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with deep supervision (multiple segments).

        Args:
            x: Input tokens [batch, seq_len]
            num_segments: Number of times to run forward_pass

        Returns:
            y_hat: Final output predictions
            state: Final hidden state
        """
        state: Optional[torch.Tensor] = None

        for _ in range(num_segments):
            state, y_hat = self.forward_pass(x, state)
            # Detach state between segments for truncated BPTT
            state = state.detach()

        return y_hat, state


class StandardTransformer(nn.Module):
    """
    Standard Transformer LM without deep supervision support.
    This serves as a baseline to show the effect of deep supervision.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        seq_len: int = 64,
        num_layers: int = 2,
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

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.

        Args:
            x: Input tokens [batch, seq_len]

        Returns:
            y_hat: Output predictions [batch, seq_len, vocab_size]
        """
        h = self.embedding(x)

        for layer in self.layers:
            h = layer(h)

        return self.output_head(h)
