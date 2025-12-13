"""
LEGO Framework - Model Components

LEGOTransformer: Unified model supporting standard and early exit modes.
"""

from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules import TransformerBlock


class LEGOTransformer(nn.Module):
    """
    LEGO Transformer for language modeling.

    Unified model supporting:
    - Standard training (final layer loss only)
    - Deep supervision (loss at all layers)
    - Early exit inference (confidence-based routing)

    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        exit_layer: Layer for early exit (None = no early exit)
        routing_threshold: Confidence threshold for early exit
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        exit_layer: Optional[int] = None,
        routing_threshold: float = 0.8,
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

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """Compute confidence (max probability) from hidden state."""
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: output from final layer."""
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)  # type: ignore[no-any-return]

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward returning output from each layer (for Deep Supervision)."""
        h = self.embedding(x)
        outputs = []
        for layer in self.layers:
            h = layer(h)
            outputs.append(self.output_head(h))
        return outputs

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

    def extend(
        self,
        num_layers: int,
        routing_threshold: float,
        freeze_lower: bool = True,
    ) -> 'LEGOTransformer':
        """
        Create an extended model from this model.

        Copies weights and optionally freezes lower layers.
        Upper layers are randomly initialized.

        Args:
            num_layers: Total number of layers for extended model
            routing_threshold: Confidence threshold for early exit
            freeze_lower: Whether to freeze lower layers (default: True)

        Returns:
            Extended LEGOTransformer with copied weights
        """
        exit_layer = self.num_layers

        # Create extended model
        extended = LEGOTransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=num_layers,
            num_heads=self.layers[0].attn.num_heads,
            exit_layer=exit_layer,
            routing_threshold=routing_threshold,
        )

        # Copy weights from this model
        extended.embedding.load_state_dict(self.embedding.state_dict())
        for i in range(self.num_layers):
            extended.layers[i].load_state_dict(self.layers[i].state_dict())
        extended.output_head.load_state_dict(self.output_head.state_dict())

        # Freeze lower layers if requested
        if freeze_lower:
            for param in extended.embedding.parameters():
                param.requires_grad = False
            for i in range(self.num_layers):
                for param in extended.layers[i].parameters():
                    param.requires_grad = False

        return extended

