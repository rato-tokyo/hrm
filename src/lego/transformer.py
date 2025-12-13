"""
LEGO Framework - LEGOTransformer

Manages LEGOBlocks and handles inter-block routing with early exit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, List, Any, TYPE_CHECKING

from .block import LEGOBlock

if TYPE_CHECKING:
    from .data import TrainingData


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

    def forward_with_routing(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with TRUE token-level early exit.

        Each token independently exits at the first block where its confidence
        exceeds the threshold. Tokens that exit early do NOT pass through
        subsequent blocks (TRUE early exit).

        Args:
            x: Input token ids (batch_size, seq_len)

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            stats: Dictionary with exit_counts, shallow_ratio, compute_cost
        """
        batch_size, seq_len = x.shape
        device = x.device

        h = self.embedding(x)
        h_flat = h.view(-1, self.dim)
        total_tokens = batch_size * seq_len

        final_logits = torch.zeros(total_tokens, self.vocab_size, device=device)
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

        final_logits = final_logits.view(batch_size, seq_len, self.vocab_size)
        exit_counts = [
            int((exit_blocks == i).sum().item()) for i in range(len(self.blocks))
        ]

        return final_logits, self._compute_exit_stats(exit_counts)

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

    def create_training_data(
        self,
        batches: List[Tuple[torch.Tensor, torch.Tensor]],
        device: str
    ) -> "TrainingData":
        """
        Create TrainingData from raw token batches.

        Converts (input_ids, target_ids) batches into (hidden_states, targets)
        by passing through the embedding layer.

        Args:
            batches: List of (input_ids, target_ids) tuples
            device: Device to run on

        Returns:
            TrainingData ready for block training
        """
        from .data import TrainingData

        all_hidden: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for x, y in batches:
                x, y = x.to(device), y.to(device)
                h = self.embedding(x)  # (batch, seq, dim)
                all_hidden.append(h.view(-1, self.dim))
                all_targets.append(y.view(-1))

        return TrainingData(
            torch.cat(all_hidden),
            torch.cat(all_targets)
        )
