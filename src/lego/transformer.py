"""
LEGO Framework - LEGOTransformer

Manages LEGOBlocks and handles inter-block routing with early exit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, List, Any, Union, overload, Literal

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
        dim: Model dimension (embedding dimension)
        blocks: List of LEGOBlock instances
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        blocks: List[LEGOBlock],
    ):
        super().__init__()

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
        dim: int,
        num_heads: int,
        layers_per_block: List[int],
        thresholds: List[float],
    ) -> 'LEGOTransformer':
        """
        Create a LEGOTransformer with multiple blocks.

        Args:
            vocab_size: Vocabulary size
            dim: Model dimension
            num_heads: Number of attention heads
            layers_per_block: Number of layers in each block
            thresholds: Early exit threshold for each block (1.0 = no early exit)

        Returns:
            LEGOTransformer with specified blocks

        Example:
            # 3 blocks: 2 layers each, thresholds 0.8, 0.9, 1.0
            model = LEGOTransformer.create(
                vocab_size=10000,
                dim=64,
                num_heads=4,
                layers_per_block=[2, 2, 2],
                thresholds=[0.8, 0.9, 1.0]
            )
        """
        if len(layers_per_block) != len(thresholds):
            raise ValueError(
                f"layers_per_block and thresholds must have same length: "
                f"{len(layers_per_block)} != {len(thresholds)}"
            )

        blocks = [
            LEGOBlock(dim, num_heads, num_layers, threshold)
            for num_layers, threshold in zip(layers_per_block, thresholds)
        ]
        return cls(vocab_size, dim, blocks)

    def _init_weights(self) -> None:
        """Initialize weights for embedding and output head."""
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.output_head.weight, std=1.0 / math.sqrt(self.embedding.embedding_dim))

    @property
    def num_layers(self) -> int:
        """Total number of layers across all blocks."""
        return sum(block.num_layers for block in self.blocks)

    @overload
    def forward(
        self, x: torch.Tensor, *, return_stats: Literal[False] = ...
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self, x: torch.Tensor, *, return_stats: Literal[True]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]: ...

    def forward(
        self, x: torch.Tensor, *, return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with TRUE token-level early exit.

        Each token independently exits at the first block where its confidence
        exceeds the threshold. Tokens that exit early do NOT pass through
        subsequent blocks (TRUE early exit).

        Args:
            x: Input token ids (batch_size, seq_len)
            return_stats: If True, return (logits, stats) tuple with exit statistics

        Returns:
            If return_stats=False: logits (batch_size, seq_len, vocab_size)
            If return_stats=True: (logits, stats) where stats contains
                exit_counts, shallow_ratio, compute_cost
        """
        batch_size, seq_len = x.shape
        device = x.device
        dim = self.embedding.embedding_dim

        h = self.embedding(x)
        h_flat = h.view(-1, dim)
        total_tokens = batch_size * seq_len

        vocab_size = self.embedding.num_embeddings
        final_logits = torch.zeros(total_tokens, vocab_size, device=device)
        exit_blocks = torch.full((total_tokens,), len(self.blocks) - 1, device=device)
        active_indices = torch.arange(total_tokens, device=device)

        for block_idx, block in enumerate(self.blocks):
            if len(active_indices) == 0:
                break

            is_last_block = (block_idx == len(self.blocks) - 1)
            h_active = h_flat[active_indices].unsqueeze(1)

            h_active, logits_active, should_exit = block.forward(h_active)
            logits_active = logits_active.squeeze(1)
            should_exit = should_exit.squeeze(1)

            if not is_last_block:
                exiting_indices = active_indices[should_exit]
                final_logits[exiting_indices] = logits_active[should_exit]
                exit_blocks[exiting_indices] = block_idx

                continuing_mask = ~should_exit
                h_flat[active_indices[continuing_mask]] = h_active.squeeze(1)[continuing_mask]
                active_indices = active_indices[continuing_mask]
            else:
                final_logits[active_indices] = logits_active

        final_logits = final_logits.view(batch_size, seq_len, vocab_size)

        if return_stats:
            exit_counts = [
                int((exit_blocks == i).sum().item()) for i in range(len(self.blocks))
            ]
            return final_logits, self._compute_exit_stats(exit_counts)

        return final_logits

    def _compute_exit_stats(self, exit_counts: List[int]) -> Dict[str, Any]:
        """Compute statistics from exit counts.

        Args:
            exit_counts: Number of tokens exiting at each block

        Returns:
            Dictionary with exit_counts, shallow_ratio, compute_cost
        """
        total_tokens = sum(exit_counts)

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

        # Shallow ratio: all exits except last block
        shallow_exits = sum(exit_counts[:-1]) if len(exit_counts) > 1 else 0
        shallow_ratio = shallow_exits / total_tokens if total_tokens > 0 else 0.0

        return {
            'exit_counts': exit_counts,
            'shallow_ratio': shallow_ratio,
            'compute_cost': compute_cost,
        }
