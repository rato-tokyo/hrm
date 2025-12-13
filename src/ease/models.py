"""
LEGO Framework - Model Components

Layered Ensemble with Gradual Optimization
"""

from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules import TransformerBlock


class LEGOTransformer(nn.Module):
    """
    LEGO Transformer for multi-phase training with optional early exit.

    A unified transformer model that supports:
    - Standard training (final layer loss)
    - Deep supervision (loss at all layers)
    - Early exit inference (confidence-based routing)
    - Multi-phase LEGO training

    Args:
        vocab_size: Vocabulary size
        dim: Model dimension (default: 64)
        num_layers: Number of transformer layers (default: 3)
        num_heads: Number of attention heads (default: 4)
        exit_layer: Layer for early exit evaluation, 1-indexed (default: None, disabled)
        routing_threshold: Confidence threshold for early exit (default: 0.5)

    Usage:
        # Standard usage
        model = LEGOTransformer(vocab_size=10000, num_layers=4)
        output = model(x)

        # With early exit
        model = LEGOTransformer(vocab_size=10000, num_layers=4, exit_layer=2)
        output, stats = model.forward_with_routing(x)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        exit_layer: Optional[int] = None,
        routing_threshold: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.exit_layer = exit_layer if exit_layer is not None else num_layers
        self.routing_threshold = routing_threshold

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: output from final layer."""
        h = self.forward_to_hidden(x)
        return self.output_head(h)  # type: ignore[no-any-return]

    def forward_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning final hidden state (before output_head).

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Hidden state tensor of shape (batch_size, seq_len, dim)
        """
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return h

    def forward_to_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Forward pass up to specified layer, returning hidden state.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            layer_idx: Layer index (1-indexed) to stop at

        Returns:
            Hidden state tensor of shape (batch_size, seq_len, dim)
        """
        h = self.embedding(x)
        for i in range(min(layer_idx, self.num_layers)):
            h = self.layers[i](h)
        return h

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward returning output from each layer (for Deep Supervision)."""
        h = self.embedding(x)
        outputs = []
        for layer in self.layers:
            h = layer(h)
            outputs.append(self.output_head(h))
        return outputs

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction confidence from hidden state.

        Args:
            h: Hidden state tensor of shape (batch_size, seq_len, dim)

        Returns:
            Confidence values of shape (batch_size, seq_len), range [0, 1]
        """
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values

    def forward_with_routing(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward with early exit routing information.

        Used by Trainer for evaluation with routing.

        Returns:
            Dictionary with:
            - shallow_logits: Output after exit_layer
            - deep_logits: Output after all layers
            - confidence: Confidence at exit point
        """
        h = self.embedding(x)

        # Process up to exit layer
        for i in range(self.exit_layer):
            h = self.layers[i](h)

        shallow_logits = self.output_head(h)

        # Compute confidence at exit point
        with torch.no_grad():
            confidence = self.compute_confidence(h)

        # Continue to deep output
        h_deep = h
        for i in range(self.exit_layer, self.num_layers):
            h_deep = self.layers[i](h_deep)
        deep_logits = self.output_head(h_deep)

        return {
            'shallow_logits': shallow_logits,
            'deep_logits': deep_logits,
            'confidence': confidence,
        }


# Backward compatibility aliases
StandardTransformer = LEGOTransformer
DeepSupervisionTransformer = LEGOTransformer
