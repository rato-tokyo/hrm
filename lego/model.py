"""
LEGO Framework - LEGOLLM

The complete LEGO language model with multi-block early exit capability.
Manages LEGOBlocks and handles inter-block routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, List, Any, Union, overload, Literal

from .block import LEGOBlock


class LEGOLLM(nn.Module):
    """
    LEGO Language Model with multi-block early exit.

    The complete model architecture:
    - Embedding layer (token -> hidden states)
    - Multiple LEGOBlocks with early exit capability
    - Shared output head (hidden states -> logits)

    Manages:
    - Inter-block routing based on confidence
    - TRUE early exit (exited tokens skip subsequent blocks)
    - Exit statistics computation

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
        Forward pass with TRUE token-level early exit (sequence-based processing).

        Each token independently exits at the first block where its confidence
        exceeds the threshold. Tokens that exit early do NOT pass through
        subsequent blocks (TRUE early exit).

        Processing is done sequence-wise to maintain proper Attention computation.
        Exit decisions are made per-token within each sequence.

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
        vocab_size = self.embedding.num_embeddings

        # Embedding
        h = self.embedding(x)  # (batch_size, seq_len, dim)

        # Initialize output tensors
        final_logits = torch.zeros(batch_size, seq_len, vocab_size, device=device)
        exit_blocks = torch.full((batch_size, seq_len), len(self.blocks) - 1, device=device)
        exited_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for block_idx, block in enumerate(self.blocks):
            is_last_block = (block_idx == len(self.blocks) - 1)

            # Process full sequences through block (for proper Attention)
            h_out, logits, should_exit, _ = block.forward(h)
            # h_out: (batch_size, seq_len, dim)
            # logits: (batch_size, seq_len, vocab_size)
            # should_exit: (batch_size, seq_len)
            # hidden_history: not used here (for analysis/training)

            if not is_last_block:
                # Tokens that should exit now (and haven't exited before)
                new_exits = should_exit & ~exited_mask

                # Store logits for newly exiting tokens
                final_logits[new_exits] = logits[new_exits]
                exit_blocks[new_exits] = block_idx

                # Update exited mask
                exited_mask = exited_mask | should_exit

                # Update hidden states for next block
                # Only update tokens that haven't exited
                h = torch.where(
                    exited_mask.unsqueeze(-1).expand_as(h_out),
                    h,  # Keep old hidden states for exited tokens
                    h_out  # Use new hidden states for continuing tokens
                )
            else:
                # Last block: store logits for all remaining tokens
                remaining_mask = ~exited_mask
                final_logits[remaining_mask] = logits[remaining_mask]

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
