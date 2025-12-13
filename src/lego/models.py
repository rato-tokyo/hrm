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

    def forward_with_cache_partial(
        self,
        h: torch.Tensor,
        past_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        start_layer: int,
        end_layer: int,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process through a range of layers with KV cache.

        Args:
            h: Hidden states (batch_size, seq_len, dim) - NOT token ids
            past_kv_cache: KV cache for layers in [start_layer, end_layer)
            start_layer: First layer to process (inclusive)
            end_layer: Last layer to process (exclusive)

        Returns:
            h: Output hidden states
            new_kv_cache: Updated KV cache for processed layers
        """
        new_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i in range(start_layer, end_layer):
            cache_idx = i - start_layer
            layer_cache = past_kv_cache[cache_idx] if past_kv_cache else None
            h, new_cache = self.layers[i](h, kv_cache=layer_cache, use_cache=True)
            new_kv_cache.append(new_cache)

        return h, new_kv_cache

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

    def generate_with_early_exit(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        routing_threshold: Optional[float] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Autoregressive generation with TRUE early exit.

        For each token:
        1. Process through lower layers (0 to exit_layer-1)
        2. Compute confidence
        3. If confidence >= threshold: use shallow output, SKIP upper layers
        4. If confidence < threshold: process through upper layers

        Args:
            input_ids: Initial token ids (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            routing_threshold: Confidence threshold (default: self.routing_threshold)
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            generated: Generated token ids
            stats: {shallow_count, deep_count, total_layers_computed, actual_compute_cost}
        """
        if routing_threshold is None:
            routing_threshold = self.routing_threshold

        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        # Separate KV caches for lower and upper layers
        lower_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        upper_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        # Statistics
        shallow_count = 0
        deep_count = 0
        total_layers_computed = 0

        with torch.no_grad():
            # Process initial prompt through ALL layers (no early exit for prompt)
            h = self.embedding(input_ids)

            # Lower layers
            h, lower_kv_cache = self.forward_with_cache_partial(
                h, None, 0, self.exit_layer
            )

            # Upper layers
            h, upper_kv_cache = self.forward_with_cache_partial(
                h, None, self.exit_layer, self.num_layers
            )

            logits = self.output_head(h)
            total_layers_computed += input_ids.shape[1] * self.num_layers

            # Generate new tokens one by one
            for _ in range(max_new_tokens):
                # Get logits for last position and sample
                next_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                # Process new token through lower layers
                h = self.embedding(next_token)
                h, new_lower_cache = self.forward_with_cache_partial(
                    h, lower_kv_cache, 0, self.exit_layer
                )
                lower_kv_cache = new_lower_cache
                total_layers_computed += self.exit_layer

                # Compute confidence at exit point
                shallow_logits = self.output_head(h)
                shallow_probs = F.softmax(shallow_logits, dim=-1)
                confidence = shallow_probs.max(dim=-1).values  # (batch, 1)

                # Early exit decision (per batch item)
                use_shallow = (confidence >= routing_threshold).all().item()

                if use_shallow:
                    # Use shallow output, SKIP upper layers
                    logits = shallow_logits
                    shallow_count += batch_size
                    # Upper cache is NOT updated (shallow tokens don't go through upper layers)
                else:
                    # Process through upper layers
                    h, new_upper_cache = self.forward_with_cache_partial(
                        h, upper_kv_cache, self.exit_layer, self.num_layers
                    )
                    upper_kv_cache = new_upper_cache
                    logits = self.output_head(h)
                    deep_count += batch_size
                    total_layers_computed += (self.num_layers - self.exit_layer)

        # Compute actual compute cost
        total_tokens = shallow_count + deep_count
        if total_tokens > 0:
            # Cost relative to always using all layers
            actual_layers = (shallow_count * self.exit_layer +
                            deep_count * self.num_layers)
            actual_compute_cost = actual_layers / (total_tokens * self.num_layers)
        else:
            actual_compute_cost = 1.0

        stats = {
            'shallow_count': shallow_count,
            'deep_count': deep_count,
            'shallow_ratio': shallow_count / max(total_tokens, 1),
            'actual_compute_cost': actual_compute_cost,
            'total_layers_computed': total_layers_computed,
        }

        return generated, stats

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

