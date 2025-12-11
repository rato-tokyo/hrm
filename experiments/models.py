"""
Shared model components for experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict

import sys
sys.path.append('..')
from modules.transformer import TransformerBlock


class StandardTransformer(nn.Module):
    """
    Standard Transformer for language modeling.

    Supports:
    - Variable number of layers
    - forward(): Standard forward (final layer output)
    - forward_all_layers(): Output from each layer (for LPT)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers

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
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward returning output from each layer (for LPT training)."""
        h = self.embedding(x)
        outputs = []
        for layer in self.layers:
            h = layer(h)
            outputs.append(self.output_head(h))
        return outputs

    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward returning final output and all hidden states."""
        h = self.embedding(x)
        hidden_states = []
        for layer in self.layers:
            h = layer(h)
            hidden_states.append(h)
        return self.output_head(h), hidden_states


# Alias for clarity
LPTTransformer = StandardTransformer


class ConfidenceRoutedTransformer(nn.Module):
    """
    Confidence-Routed Transformer.

    Routes tokens to different depth paths based on confidence at exit point.
    - Shallow path: L1 → Output (1 layer)
    - Deep path: L1 → L2 → L3 → Output (all layers)

    Architecture:
    - exit_layer: Layer after which to compute confidence (default: 1)
    - num_layers: Total number of layers (default: 3)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        exit_layer: int = 1,
        routing_threshold: float = 0.8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.exit_layer = exit_layer
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

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """Compute confidence (max probability) from hidden state."""
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward (deep path only, for compatibility)."""
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward returning output from each layer."""
        h = self.embedding(x)
        outputs = []
        for layer in self.layers:
            h = layer(h)
            outputs.append(self.output_head(h))
        return outputs

    def forward_train(self, x: torch.Tensor) -> Dict:
        """
        Training forward: compute both shallow and deep outputs.

        Returns dict with:
        - shallow_logits: Output after exit_layer
        - deep_logits: Output after all layers
        - confidence: Confidence at exit point
        - shallow_ratio: Fraction of tokens that would exit early
        """
        h = self.embedding(x)

        # Process up to exit layer
        for i in range(self.exit_layer):
            h = self.layers[i](h)

        # Shallow output (at exit point)
        shallow_logits = self.output_head(h)

        # Continue to deep output
        h_deep = h
        for i in range(self.exit_layer, self.num_layers):
            h_deep = self.layers[i](h_deep)
        deep_logits = self.output_head(h_deep)

        # Compute confidence at exit point
        with torch.no_grad():
            confidence = self.compute_confidence(h)

        return {
            'shallow_logits': shallow_logits,
            'deep_logits': deep_logits,
            'confidence': confidence,
            'shallow_ratio': (confidence >= self.routing_threshold).float().mean().item(),
        }

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Inference forward: hard routing based on confidence.

        Returns:
            output: Routed output
            stats: Dictionary with routing statistics
        """
        batch_size, seq_len = x.shape

        h = self.embedding(x)

        # Process up to exit layer
        for i in range(self.exit_layer):
            h = self.layers[i](h)

        # Compute confidence for routing
        confidence = self.compute_confidence(h)
        shallow_logits = self.output_head(h)

        # Deep path
        h_deep = h
        for i in range(self.exit_layer, self.num_layers):
            h_deep = self.layers[i](h_deep)
        deep_logits = self.output_head(h_deep)

        # Hard routing
        mask = (confidence >= self.routing_threshold).unsqueeze(-1)
        output = torch.where(mask, shallow_logits, deep_logits)

        # Compute cost
        shallow_count = mask.sum().item()
        total_count = batch_size * seq_len
        deep_count = total_count - shallow_count

        # Cost: shallow uses exit_layer layers, deep uses num_layers layers
        compute_cost = (shallow_count * self.exit_layer + deep_count * self.num_layers) / (total_count * self.num_layers)

        stats = {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': compute_cost,
        }

        return output, stats


