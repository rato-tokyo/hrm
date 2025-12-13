"""
LEGO Framework - Model Components

Layered Ensemble with Gradual Optimization: レゴブロックのようにStageを組み合わせる柔軟な訓練アーキテクチャ

Two base models:
- StandardTransformer: Final layer loss only
- DeepSupervisionTransformer: Loss at all layers with early exit support

Both support LEGO's 2 core options:
- stages: Stage-based training configuration (LEGO blocks)
- routing_threshold: Early exit at inference
"""

from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules import TransformerBlock


class BaseTransformer(nn.Module):
    """
    Base Transformer class with shared functionality.

    Contains common components:
    - Embedding layer
    - Transformer layers
    - Output head
    - Weight initialization
    - forward(), forward_to_hidden(), forward_all_layers() methods
    - compute_confidence() for confidence-based routing
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

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction confidence from hidden state.

        Confidence is defined as the maximum probability in the softmax distribution.
        Higher confidence indicates the model is more certain about its prediction.

        Args:
            h: Hidden state tensor of shape (batch_size, seq_len, dim)

        Returns:
            Confidence values of shape (batch_size, seq_len), range [0, 1]
        """
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: output from final layer."""
        h = self.forward_to_hidden(x)
        return self.output_head(h)  # type: ignore[no-any-return]

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward returning output from each layer (for Deep Supervision)."""
        h = self.embedding(x)
        outputs = []
        for layer in self.layers:
            h = layer(h)
            outputs.append(self.output_head(h))
        return outputs

    def forward_to_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Forward pass up to specified layer, returning hidden state.

        Used in multi-phase LEGO training to compute confidence at intermediate layers.

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


class StandardTransformer(BaseTransformer):
    """
    Standard Transformer for language modeling.

    Base model with final layer loss only.
    Supports forward_all_layers() for deep supervision training.
    """
    pass


class DeepSupervisionTransformer(BaseTransformer):
    """
    Deep Supervision Transformer with Early Exit support.

    Training: Loss at all layers (configurable weights via TrainingConfig).
    Inference: Optional confidence-based early exit.

    References:
    - Deep Supervision: Lee et al., 2015
    - Early Exit: Teerapittayanon et al., 2016
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
        super().__init__(vocab_size, dim, num_layers, num_heads)
        self.exit_layer = exit_layer
        self.routing_threshold = routing_threshold

    def forward_train(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
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

        compute_cost = (shallow_count * self.exit_layer + deep_count * self.num_layers) / (total_count * self.num_layers)

        stats = {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': compute_cost,
        }

        return output, stats
