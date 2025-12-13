"""
LEGO Framework - Model Components

LEGOBlock: Block definition with layer range and threshold.
LEGOTransformer: Unified model supporting multi-block early exit.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules import TransformerBlock
from .utils import compute_routing_cost


@dataclass
class LEGOBlock:
    """
    A block of transformer layers with an exit threshold.

    Args:
        start_layer: First layer index (inclusive)
        end_layer: Last layer index (exclusive)
        threshold: Confidence threshold for early exit after this block.
                   Use 1.0 for the final block (no early exit possible).
    """
    start_layer: int
    end_layer: int
    threshold: float = 1.0

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer


class LEGOTransformer(nn.Module):
    """
    LEGO Transformer for language modeling with multi-block early exit.

    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        blocks: List of LEGOBlock defining block boundaries and thresholds.
                If None, creates a single block covering all layers.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        blocks: Optional[List[LEGOBlock]] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Default: single block covering all layers
        if blocks is None:
            blocks = [LEGOBlock(0, num_layers, threshold=1.0)]
        self.blocks = blocks

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
        Process through a range of layers with KV cache.

        Args:
            h: Hidden states (batch_size, seq_len, dim)
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
    ) -> Tuple[torch.Tensor, List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Process initial prompt through all blocks, returning logits and KV caches."""
        h = self.embedding(input_ids)

        block_caches: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        for block in self.blocks:
            h, cache = self._forward_with_cache_partial(h, None, block.start_layer, block.end_layer)
            block_caches.append(cache)

        logits = self.output_head(h)
        return logits, block_caches

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Autoregressive generation with TRUE early exit.

        For each token, processes through blocks sequentially.
        After each block (except the last), checks confidence.
        If confidence >= block.threshold, uses current output and skips remaining blocks.

        Args:
            input_ids: Initial token ids (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            generated: Generated token ids
            stats: Dictionary with exit counts per block and compute cost
        """
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        # Track exits per block
        exit_counts = [0] * len(self.blocks)

        with torch.no_grad():
            logits, block_caches = self._process_prompt(input_ids)

            for _ in range(max_new_tokens):
                next_token = self._sample_next_token(logits, temperature, top_k)
                generated = torch.cat([generated, next_token], dim=1)

                h = self.embedding(next_token)
                exited = False

                for block_idx, block in enumerate(self.blocks):
                    h, block_caches[block_idx] = self._forward_with_cache_partial(
                        h, block_caches[block_idx], block.start_layer, block.end_layer
                    )

                    # Check early exit (skip for last block)
                    if block_idx < len(self.blocks) - 1:
                        block_logits, confidence = self.compute_confidence(h)
                        if (confidence >= block.threshold).all().item():
                            logits = block_logits
                            exit_counts[block_idx] += batch_size
                            exited = True
                            break

                if not exited:
                    logits = self.output_head(h)
                    exit_counts[-1] += batch_size

        # Compute statistics
        total_tokens = sum(exit_counts)
        stats = self._compute_generate_stats(exit_counts, total_tokens)

        return generated, stats

    def _compute_generate_stats(
        self,
        exit_counts: List[int],
        total_tokens: int
    ) -> Dict[str, Any]:
        """Compute generation statistics from exit counts."""
        # Compute weighted layer cost
        total_layers_computed = 0
        for block_idx, count in enumerate(exit_counts):
            layers_for_this_exit = self.blocks[block_idx].end_layer
            total_layers_computed += count * layers_for_this_exit

        actual_compute_cost = total_layers_computed / (total_tokens * self.num_layers) if total_tokens > 0 else 1.0

        stats: Dict[str, Any] = {
            'exit_counts': exit_counts,
            'total_tokens': total_tokens,
            'actual_compute_cost': actual_compute_cost,
        }

        # For backward compatibility with 2-block case
        if len(self.blocks) == 2:
            stats['shallow_count'] = exit_counts[0]
            stats['deep_count'] = exit_counts[1]
            stats['shallow_ratio'] = exit_counts[0] / max(total_tokens, 1)

        return stats

    def forward_upper_layers(
        self,
        h: torch.Tensor,
        start_layer: int
    ) -> torch.Tensor:
        """
        Forward through layers starting from start_layer (for block training).

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
        routing_threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with routing statistics (for evaluation).

        Uses block thresholds by default, or override with routing_threshold.
        For evaluation metrics only - use generate() for actual computation savings.

        Args:
            x: Input token ids (batch_size, seq_len)
            routing_threshold: Override threshold (uses first block's threshold if None)

        Returns:
            output: Routed logits (batch_size, seq_len, vocab_size)
            stats: Dictionary with mean_confidence, shallow_ratio, compute_cost
        """
        if len(self.blocks) < 2:
            # No early exit possible with single block
            logits = self.forward(x)
            return logits, {'mean_confidence': 1.0, 'shallow_ratio': 0.0, 'compute_cost': 1.0}

        batch_size, seq_len = x.shape
        threshold = routing_threshold if routing_threshold is not None else self.blocks[0].threshold

        # Process first block
        h = self.embedding(x)
        h = self._forward_layers(h, self.blocks[0].start_layer, self.blocks[0].end_layer)
        shallow_logits, confidence = self.compute_confidence(h)

        # Process remaining blocks for deep output
        for block in self.blocks[1:]:
            h = self._forward_layers(h, block.start_layer, block.end_layer)
        deep_logits = self.output_head(h)

        # Route based on confidence
        mask = (confidence >= threshold).unsqueeze(-1)
        output = torch.where(mask, shallow_logits, deep_logits)

        # Statistics
        shallow_count = int(mask.sum().item())
        total_count = batch_size * seq_len
        deep_count = total_count - shallow_count
        first_block_layers = self.blocks[0].end_layer
        cost = compute_routing_cost(shallow_count, deep_count, first_block_layers, self.num_layers)

        return output, {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': cost,
        }

    def _freeze_layers(self, num_layers_to_freeze: int) -> None:
        """Freeze embedding and specified number of layers."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for i in range(num_layers_to_freeze):
            for param in self.layers[i].parameters():
                param.requires_grad = False

    def extend(
        self,
        num_new_layers: int,
        threshold: float,
        freeze_existing: bool = True,
    ) -> 'LEGOTransformer':
        """
        Create an extended model by adding a new block.

        Existing blocks are preserved. A new block is added with the new layers.

        Args:
            num_new_layers: Number of layers to add
            threshold: Confidence threshold for the NEW last-but-one block
                       (current last block gets this threshold, new block gets 1.0)
            freeze_existing: Whether to freeze existing layers (default: True)

        Returns:
            Extended LEGOTransformer with new block added
        """
        new_total_layers = self.num_layers + num_new_layers

        # Update existing blocks: set threshold on current last block
        new_blocks = []
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                # Last block now gets the threshold (becomes exit point)
                new_blocks.append(LEGOBlock(block.start_layer, block.end_layer, threshold))
            else:
                new_blocks.append(block)

        # Add new block
        new_blocks.append(LEGOBlock(self.num_layers, new_total_layers, threshold=1.0))

        # Create extended model
        extended = LEGOTransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=new_total_layers,
            num_heads=self.num_heads,
            blocks=new_blocks,
        )

        # Copy weights from this model
        extended.embedding.load_state_dict(self.embedding.state_dict())
        for i in range(self.num_layers):
            extended.layers[i].load_state_dict(self.layers[i].state_dict())
        extended.output_head.load_state_dict(self.output_head.state_dict())

        # Freeze existing layers if requested
        if freeze_existing:
            extended._freeze_layers(self.num_layers)

        return extended
