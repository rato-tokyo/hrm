"""
LEGO Framework - Model Components

LEGOTransformer: Unified model supporting standard and early exit modes.
"""

from typing import Tuple, Dict, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules import TransformerBlock


class LEGOTransformer(nn.Module):
    """
    LEGO Transformer for language modeling.

    Supports standard training and early exit inference (confidence-based routing).

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

    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden states after all layers (before output head)."""
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return h

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """Compute confidence (max probability) from hidden state."""
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: output from final layer."""
        h = self.get_hidden_states(x)
        return self.output_head(h)  # type: ignore[no-any-return]

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass with optional KV cache for autoregressive generation.

        Args:
            x: Input token ids (batch_size, seq_len)
            past_kv_cache: List of (K, V) tuples for each layer
            use_cache: Whether to return updated cache

        Returns:
            If use_cache=False: logits (batch_size, seq_len, vocab_size)
            If use_cache=True: (logits, new_kv_cache)
        """
        h = self.embedding(x)

        new_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_cache[i] if past_kv_cache else None
            if use_cache:
                h, new_cache = layer(h, kv_cache=layer_cache, use_cache=True)
                new_kv_cache.append(new_cache)
            else:
                h = layer(h, kv_cache=layer_cache, use_cache=False)

        logits = self.output_head(h)

        if use_cache:
            return logits, new_kv_cache
        return logits  # type: ignore[return-value]

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with KV cache.

        Args:
            input_ids: Initial token ids (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token ids (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        generated = input_ids.clone()
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        with torch.no_grad():
            # Process initial prompt
            logits, kv_cache = self.forward_with_cache(
                input_ids, past_kv_cache=None, use_cache=True
            )

            for _ in range(max_new_tokens):
                # Get logits for last position
                next_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')

                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)

                # Forward with cache (only new token)
                logits, kv_cache = self.forward_with_cache(
                    next_token, past_kv_cache=kv_cache, use_cache=True
                )

        return generated

    def _forward_early_exit(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Core early exit forward pass.

        Returns:
            shallow_logits: Output after exit_layer
            deep_logits: Output after all layers
            confidence: Confidence at exit point
        """
        h = self.embedding(x)

        # Process up to exit layer
        for i in range(self.exit_layer):
            h = self.layers[i](h)

        # Shallow output and confidence (reuse logits for efficiency)
        shallow_logits = self.output_head(h)
        probs = F.softmax(shallow_logits, dim=-1)
        confidence = probs.max(dim=-1).values

        # Continue to deep output
        h_deep = h
        for i in range(self.exit_layer, self.num_layers):
            h_deep = self.layers[i](h_deep)
        deep_logits = self.output_head(h_deep)

        return shallow_logits, deep_logits, confidence

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Inference forward: hard routing based on confidence.

        Returns:
            output: Routed output
            stats: Dictionary with routing statistics
        """
        batch_size, seq_len = x.shape

        shallow_logits, deep_logits, confidence = self._forward_early_exit(x)

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

    def forward_upper_layers(
        self,
        h: torch.Tensor,
        start_layer: int
    ) -> torch.Tensor:
        """
        Forward through upper layers only (for Phase 2 training).

        Args:
            h: Hidden states from lower layers (batch_size, seq_len, dim)
            start_layer: Index of first upper layer to process

        Returns:
            Logits after processing through upper layers
        """
        for i in range(start_layer, self.num_layers):
            h = self.layers[i](h)
        return self.output_head(h)  # type: ignore[no-any-return]

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

