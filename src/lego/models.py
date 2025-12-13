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
from .utils import compute_routing_cost


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

    def _forward_layers(self, h: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Process hidden states through layers[start:end]."""
        for i in range(start, end):
            h = self.layers[i](h)
        return h

    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Get hidden states after all layers (before output head)."""
        h = self.embedding(x)
        return self._forward_layers(h, 0, self.num_layers)

    def compute_confidence(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute confidence (max probability) from hidden state.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            logits: Output logits
            confidence: Max probability per token (batch_size, seq_len)
        """
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return logits, confidence

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

    def _forward_with_cache_partial(
        self,
        h: torch.Tensor,
        past_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        start_layer: int,
        end_layer: int,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process through a range of layers with KV cache (internal method).

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

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int]
    ) -> torch.Tensor:
        """Sample next token from logits with temperature and optional top-k filtering."""
        next_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _process_prompt(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
        List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Process initial prompt through all blocks, returning logits and KV caches."""
        h = self.embedding(input_ids)

        # Block 1
        h, block1_cache = self._forward_with_cache_partial(h, None, 0, self.exit_layer)

        # Block 2
        h, block2_cache = self._forward_with_cache_partial(h, None, self.exit_layer, self.num_layers)

        logits = self.output_head(h)
        return logits, block1_cache, block2_cache

    def generate(
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
        1. Process through Block 1 (layers 0 to exit_layer-1)
        2. Compute confidence
        3. If confidence >= threshold: use Block 1 output, SKIP Block 2
        4. If confidence < threshold: process through Block 2

        Set routing_threshold >= 1.0 to disable early exit (all tokens go deep).

        Args:
            input_ids: Initial token ids (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            routing_threshold: Confidence threshold (default: self.routing_threshold)
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            generated: Generated token ids
            stats: {shallow_count, deep_count, shallow_ratio, actual_compute_cost}
        """
        if routing_threshold is None:
            routing_threshold = self.routing_threshold

        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        shallow_count = 0
        deep_count = 0

        with torch.no_grad():
            logits, block1_cache, block2_cache = self._process_prompt(input_ids)

            for _ in range(max_new_tokens):
                next_token = self._sample_next_token(logits, temperature, top_k)
                generated = torch.cat([generated, next_token], dim=1)

                # Block 1
                h = self.embedding(next_token)
                h, block1_cache = self._forward_with_cache_partial(
                    h, block1_cache, 0, self.exit_layer
                )

                # Confidence check
                shallow_logits, confidence = self.compute_confidence(h)
                use_shallow = (confidence >= routing_threshold).all().item()

                if use_shallow:
                    # Early exit: skip Block 2
                    logits = shallow_logits
                    shallow_count += batch_size
                else:
                    # Block 2
                    h, block2_cache = self._forward_with_cache_partial(
                        h, block2_cache, self.exit_layer, self.num_layers
                    )
                    logits = self.output_head(h)
                    deep_count += batch_size

        total_tokens = shallow_count + deep_count
        return generated, {
            'shallow_count': shallow_count,
            'deep_count': deep_count,
            'shallow_ratio': shallow_count / max(total_tokens, 1),
            'actual_compute_cost': compute_routing_cost(
                shallow_count, deep_count, self.exit_layer, self.num_layers
            ),
        }

    def forward_upper_layers(
        self,
        h: torch.Tensor,
        start_layer: int
    ) -> torch.Tensor:
        """
        Forward through layers starting from start_layer (for Block 2+ training).

        Args:
            h: Hidden states from previous block (batch_size, seq_len, dim)
            start_layer: Index of first layer to process

        Returns:
            Logits after processing through remaining layers
        """
        h = self._forward_layers(h, start_layer, self.num_layers)
        return self.output_head(h)  # type: ignore[no-any-return]

    def forward_with_routing(
        self,
        x: torch.Tensor,
        routing_threshold: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with routing statistics (for evaluation).

        Computes both Block 1 and Block 2 outputs, then routes based on confidence.
        This is for evaluation metrics only - for actual inference with
        computation savings, use generate().

        Args:
            x: Input token ids (batch_size, seq_len)
            routing_threshold: Confidence threshold for routing

        Returns:
            output: Routed logits (batch_size, seq_len, vocab_size)
            stats: Dictionary with mean_confidence, shallow_ratio, compute_cost
        """
        batch_size, seq_len = x.shape

        # Block 1
        h = self.embedding(x)
        h = self._forward_layers(h, 0, self.exit_layer)
        shallow_logits, confidence = self.compute_confidence(h)

        # Block 2
        h_deep = self._forward_layers(h, self.exit_layer, self.num_layers)
        deep_logits = self.output_head(h_deep)

        # Route based on confidence
        mask = (confidence >= routing_threshold).unsqueeze(-1)
        output = torch.where(mask, shallow_logits, deep_logits)

        # Statistics
        shallow_count = int(mask.sum().item())
        total_count = batch_size * seq_len
        deep_count = total_count - shallow_count
        cost = compute_routing_cost(shallow_count, deep_count, self.exit_layer, self.num_layers)

        return output, {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': cost,
        }

    def _freeze_layers(self, num_layers_to_freeze: int) -> None:
        """Freeze embedding and specified number of lower layers."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for i in range(num_layers_to_freeze):
            for param in self.layers[i].parameters():
                param.requires_grad = False

    def extend(
        self,
        num_layers: int,
        routing_threshold: float,
        freeze_lower: bool = True,
    ) -> 'LEGOTransformer':
        """
        Create an extended model by adding a new block.

        Current model becomes Block 1 (weights copied, optionally frozen).
        New layers become Block 2 (randomly initialized).

        Args:
            num_layers: Total number of layers for extended model
            routing_threshold: Confidence threshold for early exit
            freeze_lower: Whether to freeze Block 1 (default: True)

        Returns:
            Extended LEGOTransformer
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
            extended._freeze_layers(self.num_layers)

        return extended
