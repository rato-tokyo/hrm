"""
LEGO Framework - LEGOTransformer

Manages LEGOBlocks and handles inter-block routing with early exit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # Used in _sample_next_token
import math
from typing import Tuple, Dict, Optional, List, Any

from .block import LEGOBlock


class LEGOTransformer(nn.Module):
    """
    LEGO Transformer for language modeling with multi-block early exit.

    Manages LEGOBlocks and handles:
    - Embedding and output head (shared across blocks)
    - Inter-block routing based on confidence
    - Hard token collection and forwarding to next block

    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        num_heads: Number of attention heads
        blocks: List of LEGOBlock instances
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        blocks: List[LEGOBlock],
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads

        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(blocks)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Share output_head reference with all blocks
        for block in self.blocks:
            block.set_output_head(self.output_head)

        self._init_weights()

    @classmethod
    def create(
        cls,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        threshold: float = 1.0
    ) -> 'LEGOTransformer':
        """
        Create a LEGOTransformer with a single block.

        Args:
            vocab_size: Vocabulary size
            dim: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            threshold: Early exit threshold (1.0 = no early exit)

        Returns:
            LEGOTransformer with single block
        """
        block = LEGOBlock(dim, num_heads, num_layers, threshold)
        return cls(vocab_size, dim, num_heads, [block])

    def _init_weights(self) -> None:
        """Initialize weights for embedding and output head."""
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.output_head.weight, std=1.0 / math.sqrt(self.dim))

    @property
    def num_layers(self) -> int:
        """Total number of layers across all blocks."""
        return sum(block.num_layers for block in self.blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: process through all blocks."""
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.output_head(h)  # type: ignore[no-any-return]

    def get_hidden_states(self, x: torch.Tensor, up_to_block: int) -> torch.Tensor:
        """
        Get hidden states after specified block.

        Args:
            x: Input token ids (batch_size, seq_len)
            up_to_block: Block index to stop at (inclusive, 0-indexed). Must be explicit.

        Returns:
            Hidden states after the specified block
        """
        h = self.embedding(x)
        for block in self.blocks[:up_to_block + 1]:
            h = block(h)
        return h

    def forward_from_block(self, h: torch.Tensor, start_block_idx: int) -> torch.Tensor:
        """
        Forward through blocks starting from start_block_idx.

        Used for training new blocks on hard examples.

        Args:
            h: Hidden states from previous block (batch_size, seq_len, dim)
            start_block_idx: Index of first block to process

        Returns:
            Logits after processing through remaining blocks
        """
        for block in self.blocks[start_block_idx:]:
            h = block(h)
        return self.output_head(h)  # type: ignore[no-any-return]

    def forward_with_routing(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with TRUE token-level early exit.

        Each token independently exits at the first block where its confidence
        exceeds the threshold. Tokens that exit early do not pass through
        subsequent blocks.

        Args:
            x: Input token ids (batch_size, seq_len)

        Returns:
            output: Logits (batch_size, seq_len, vocab_size)
            stats: Dictionary with exit_counts, shallow_ratio, compute_cost
        """
        batch_size, seq_len = x.shape
        device = x.device

        h = self.embedding(x)

        # Track which tokens have exited and their final logits
        # Shape: (batch_size, seq_len, vocab_size)
        final_logits = torch.zeros(
            batch_size, seq_len, self.vocab_size, device=device
        )
        # Track exit block for each token: (batch_size, seq_len)
        exit_blocks = torch.full(
            (batch_size, seq_len), len(self.blocks) - 1, device=device
        )
        # Track which tokens are still active (not yet exited)
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        for block_idx, block in enumerate(self.blocks):
            is_last_block = (block_idx == len(self.blocks) - 1)

            # Process all tokens through this block
            h = block(h)
            logits, confidence = block.compute_confidence(h)

            if not is_last_block:
                # Determine which active tokens should exit at this block
                should_exit_here = active_mask & (confidence >= block.threshold)

                # Store logits for exiting tokens
                final_logits[should_exit_here] = logits[should_exit_here]
                exit_blocks[should_exit_here] = block_idx

                # Update active mask
                active_mask = active_mask & ~should_exit_here
            else:
                # Last block: all remaining active tokens exit here
                final_logits[active_mask] = logits[active_mask]

        # Compute statistics
        exit_counts = [
            int((exit_blocks == i).sum().item()) for i in range(len(self.blocks))
        ]
        total_tokens = batch_size * seq_len
        shallow_exits = sum(exit_counts[:-1])  # All except last block
        shallow_ratio = shallow_exits / total_tokens if total_tokens > 0 else 0.0

        # Compute weighted layer cost
        total_layers_computed = 0
        layers_so_far = 0
        for block_idx, count in enumerate(exit_counts):
            layers_so_far += self.blocks[block_idx].num_layers
            total_layers_computed += count * layers_so_far

        compute_cost = (
            total_layers_computed / (total_tokens * self.num_layers)
            if total_tokens > 0 else 1.0
        )

        return final_logits, {
            'exit_counts': exit_counts,
            'shallow_ratio': shallow_ratio,
            'compute_cost': compute_cost,
        }

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
        If confidence >= threshold, uses current output and skips remaining blocks.

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

        exit_counts = [0] * len(self.blocks)

        with torch.no_grad():
            # Process prompt through all blocks
            logits, block_caches = self._process_prompt(input_ids)

            for _ in range(max_new_tokens):
                next_token = self._sample_next_token(logits, temperature, top_k)
                generated = torch.cat([generated, next_token], dim=1)

                h = self.embedding(next_token)
                exited = False

                for block_idx, block in enumerate(self.blocks):
                    h, block_caches[block_idx] = block.forward_with_cache(
                        h, block_caches[block_idx]
                    )

                    # Check early exit (skip for last block)
                    if block_idx < len(self.blocks) - 1:
                        block_logits, confidence = block.compute_confidence(h)
                        if block.should_exit(confidence):
                            logits = block_logits
                            exit_counts[block_idx] += batch_size
                            exited = True
                            break

                if not exited:
                    logits = self.output_head(h)
                    exit_counts[-1] += batch_size

        stats = self._compute_generate_stats(exit_counts)
        return generated, stats

    def _process_prompt(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Process initial prompt through all blocks, returning logits and KV caches."""
        h = self.embedding(input_ids)

        block_caches: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        for block in self.blocks:
            h, cache = block.forward_with_cache(h, None)
            block_caches.append(cache)

        logits = self.output_head(h)
        return logits, block_caches

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int]
    ) -> torch.Tensor:
        """Sample next token from logits."""
        next_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _compute_generate_stats(self, exit_counts: List[int]) -> Dict[str, Any]:
        """Compute generation statistics from exit counts."""
        total_tokens = sum(exit_counts)

        # Compute weighted layer cost
        total_layers_computed = 0
        layers_so_far = 0
        for block_idx, count in enumerate(exit_counts):
            layers_so_far += self.blocks[block_idx].num_layers
            total_layers_computed += count * layers_so_far

        actual_compute_cost = (
            total_layers_computed / (total_tokens * self.num_layers)
            if total_tokens > 0 else 1.0
        )

        shallow_ratio = (
            exit_counts[0] / max(total_tokens, 1)
            if len(self.blocks) >= 2 else 0.0
        )

        return {
            'exit_counts': exit_counts,
            'total_tokens': total_tokens,
            'actual_compute_cost': actual_compute_cost,
            'shallow_ratio': shallow_ratio,
        }

    def extend(
        self,
        num_new_layers: int,
        threshold: float,
        freeze_existing: bool = True,
    ) -> 'LEGOTransformer':
        """
        Create an extended model by adding a new block.

        Args:
            num_new_layers: Number of layers in the new block
            threshold: Confidence threshold for the current last block
            freeze_existing: Whether to freeze existing blocks

        Returns:
            Extended LEGOTransformer with new block added
        """
        # Update threshold on current last block
        self.blocks[-1].threshold = threshold

        # Create new block
        new_block = LEGOBlock(
            self.dim, self.num_heads, num_new_layers, threshold=1.0
        )

        # Create new model with all blocks
        new_blocks = list(self.blocks) + [new_block]
        extended = LEGOTransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_heads=self.num_heads,
            blocks=new_blocks,
        )

        # Copy weights
        extended.embedding.load_state_dict(self.embedding.state_dict())
        for i, block in enumerate(self.blocks):
            extended.blocks[i].load_state_dict(block.state_dict())
        extended.output_head.load_state_dict(self.output_head.state_dict())

        # Freeze existing if requested
        if freeze_existing:
            for param in extended.embedding.parameters():
                param.requires_grad = False
            for block in extended.blocks[:-1]:
                block.freeze()

        return extended
