"""
EASE Framework - Model Components

Efficient Asymmetric Supervision for Early-Exit Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict

from .modules import TransformerBlock


class StandardTransformer(nn.Module):
    """
    Standard Transformer for language modeling.

    Supports:
    - Variable number of layers
    - forward(): Standard forward (final layer output)
    - forward_all_layers(): Output from each layer (for Deep Supervision)
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
        """Forward returning output from each layer (for Deep Supervision)."""
        h = self.embedding(x)
        outputs = []
        for layer in self.layers:
            h = layer(h)
            outputs.append(self.output_head(h))
        return outputs


class DEEDTransformer(nn.Module):
    """
    DEED (Deep Supervision + Dynamic Early Exit) Transformer.

    Training: All tokens contribute to losses at multiple layers with α-weighted distribution.
    Inference: Confidence-based early exit routing.

    Based on DEED (Tang et al., 2023) and Deep Supervision (Lee et al., 2015).
    Loss = α × Loss_shallow + (1-α) × Loss_deep

    Key characteristic:
    - Training: ALL tokens pass through ALL layers, losses computed at specified layers
    - Inference: Tokens route based on confidence threshold

    Use UniversalTrainer with layer_weights to configure α distribution.

    References:
    - DEED: Dynamic Early Exit on Decoder (Tang et al., 2023)
    - Deeply-Supervised Nets (Lee et al., 2015)
    - Depth-Adaptive Transformer (Elbayad et al., 2020)
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
        """Forward returning output from each layer (for EASE training)."""
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

        # Cost: shallow uses exit_layer layers, deep uses num_layers layers
        compute_cost = (shallow_count * self.exit_layer + deep_count * self.num_layers) / (total_count * self.num_layers)

        stats = {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': compute_cost,
        }

        return output, stats


class MoDTransformer(nn.Module):
    """
    Mixture-of-Depths Transformer.

    Implements dynamic compute allocation where each routing layer
    selects top-k tokens for full processing, while others skip via residual.

    Based on: "Mixture-of-Depths: Dynamically allocating compute in
    transformer-based language models" (Raposo et al., 2024)

    Key features:
    - Every other block is a routing block (12.5% capacity by default)
    - Top-k selection for tokens to process
    - Static computation graph (k is fixed)
    - Auxiliary predictor for causal inference
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 4,
        capacity: float = 0.125,
        route_every_n: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.capacity = capacity
        self.route_every_n = route_every_n

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Routers: one per routing layer
        self.routers = nn.ModuleDict()
        for i in range(num_layers):
            if self._is_routing_layer(i):
                self.routers[str(i)] = nn.Linear(dim, 1, bias=False)

        # Auxiliary predictor for causal inference
        self.predictors = nn.ModuleDict()
        for i in range(num_layers):
            if self._is_routing_layer(i):
                self.predictors[str(i)] = nn.Sequential(
                    nn.Linear(dim, dim // 4),
                    nn.ReLU(),
                    nn.Linear(dim // 4, 1),
                )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def _is_routing_layer(self, layer_idx: int) -> bool:
        """Check if this layer uses routing (every n-th layer)."""
        return (layer_idx % self.route_every_n) == 0

    def _get_top_k(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Get top-k mask from router scores."""
        # scores: [batch, seq_len]
        _, indices = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward (full processing, no routing)."""
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)

    def forward_train(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Training forward with MoD routing.

        Returns dict with:
        - logits: Final output logits
        - aux_loss: Auxiliary loss for predictor training
        - routing_stats: Statistics about routing decisions
        """
        batch_size, seq_len = x.shape
        k = max(1, int(seq_len * self.capacity))

        h = self.embedding(x)

        total_routed = 0
        total_possible = 0
        aux_loss = torch.tensor(0.0, device=x.device)
        num_routing_layers = 0

        for i, layer in enumerate(self.layers):
            if self._is_routing_layer(i):
                # Compute router scores
                router_scores = self.routers[str(i)](h).squeeze(-1)  # [batch, seq_len]

                # Top-k selection (non-causal, for training)
                top_k_mask = self._get_top_k(router_scores, k)  # [batch, seq_len]

                # Auxiliary predictor (for inference)
                with torch.no_grad():
                    predictor_input = h.detach()
                predictor_logits = self.predictors[str(i)](predictor_input).squeeze(-1)

                # Auxiliary loss: BCE to predict top-k membership
                targets = top_k_mask.float()
                aux_loss = aux_loss + F.binary_cross_entropy_with_logits(
                    predictor_logits, targets
                )
                num_routing_layers += 1

                # Process only top-k tokens
                # Expand mask for hidden dim
                mask_expanded = top_k_mask.unsqueeze(-1)  # [batch, seq_len, 1]

                # Full layer output
                h_full = layer(h)

                # Weighted combination: routed tokens get layer output, others keep h
                # Multiply by router weight for gradient flow
                router_weight = torch.sigmoid(router_scores).unsqueeze(-1)
                h = torch.where(
                    mask_expanded,
                    router_weight * h_full + (1 - router_weight) * h,  # Selected: blend
                    h  # Not selected: residual
                )

                total_routed += int(top_k_mask.sum().item())
                total_possible += batch_size * seq_len
            else:
                # Full capacity layer: process all tokens
                h = layer(h)

        logits = self.output_head(h)

        # Normalize aux_loss
        if num_routing_layers > 0:
            aux_loss = aux_loss / num_routing_layers

        routing_ratio = total_routed / total_possible if total_possible > 0 else 0.0

        return {
            'logits': logits,
            'aux_loss': aux_loss,
            'routing_ratio': routing_ratio,
            'k': k,
        }

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Inference forward using predictor (causal).

        Uses the auxiliary predictor instead of top-k for causal inference.
        """
        batch_size, seq_len = x.shape

        h = self.embedding(x)

        total_routed = 0
        total_possible = 0

        for i, layer in enumerate(self.layers):
            if self._is_routing_layer(i):
                # Use predictor for causal routing decision
                predictor_logits = self.predictors[str(i)](h).squeeze(-1)
                route_mask = (torch.sigmoid(predictor_logits) > 0.5)

                mask_expanded = route_mask.unsqueeze(-1)

                h_full = layer(h)
                h = torch.where(mask_expanded, h_full, h)

                total_routed += int(route_mask.sum().item())
                total_possible += batch_size * seq_len
            else:
                h = layer(h)

        logits = self.output_head(h)

        routing_ratio = total_routed / total_possible if total_possible > 0 else 0.0

        # Estimate compute cost
        num_routing_layers = sum(
            1 for i in range(self.num_layers) if self._is_routing_layer(i)
        )
        num_full_layers = self.num_layers - num_routing_layers
        # Routing layers: only routing_ratio processed, full layers: 100%
        compute_cost = (
            num_full_layers + num_routing_layers * routing_ratio
        ) / self.num_layers

        stats = {
            'routing_ratio': routing_ratio,
            'compute_cost': compute_cost,
        }

        return logits, stats


class TokenRoutedTransformer(nn.Module):
    """
    Token-Level Routed Transformer (Mask-based Token Routing).

    Training: Each token contributes to ONLY its routed path's loss.
    - High confidence tokens → shallow loss only
    - Low confidence tokens → deep loss only

    No α parameter needed - routing decision determines which loss to use.

    Key difference from DEEDTransformer:
    - DEEDTransformer: All tokens → both losses (weighted by α)
    - TokenRoutedTransformer: Each token → one loss (determined by routing)

    Requires curriculum learning (warmup with joint training) to avoid cold-start.
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

    def forward_train(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Training forward with token-level routing.

        Returns dict with:
        - shallow_logits: Output after exit_layer
        - deep_logits: Output after all layers
        - exit_mask: Boolean mask indicating which tokens exit early
        - confidence: Confidence at exit point
        - shallow_ratio: Fraction of tokens that exit early
        """
        h = self.embedding(x)

        # Process up to exit layer
        for i in range(self.exit_layer):
            h = self.layers[i](h)

        # Compute confidence and routing decision
        shallow_logits = self.output_head(h)
        with torch.no_grad():
            probs = F.softmax(shallow_logits, dim=-1)
            confidence = probs.max(dim=-1).values
            exit_mask = (confidence >= self.routing_threshold)  # [batch, seq_len]

        # Continue to deep output (all tokens, for simplicity)
        h_deep = h
        for i in range(self.exit_layer, self.num_layers):
            h_deep = self.layers[i](h_deep)
        deep_logits = self.output_head(h_deep)

        shallow_ratio = exit_mask.float().mean().item()

        return {
            'shallow_logits': shallow_logits,
            'deep_logits': deep_logits,
            'exit_mask': exit_mask,
            'confidence': confidence,
            'shallow_ratio': shallow_ratio,
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
