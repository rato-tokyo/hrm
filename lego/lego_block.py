"""
LEGO Framework - LEGOBlock

TransformerBlock wrapper with early exit capability.
Exit decision is made by exit_fn using hidden_history from all layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, TYPE_CHECKING

from .modules import TransformerBlock
from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim

if TYPE_CHECKING:
    from .sequence_data import SequenceData


class LEGOBlock(nn.Module):
    """
    TransformerBlock with early exit capability.

    Wraps a standard TransformerBlock and adds:
    - Exit decision via exit_fn (default: CALM-style cos_sim)
    - Shared output_head for logits computation
    - Access to hidden_history for flexible exit criteria
    - Hard example collection after training

    Args:
        transformer: TransformerBlock to wrap
        exit_fn: Optional custom exit function. If None, uses default_exit_fn (CALM-style)
    """

    def __init__(
        self,
        transformer: TransformerBlock,
        exit_fn: Optional[ExitFn] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.exit_fn = exit_fn or default_exit_fn
        self.threshold = 0.0  # Set by collect_hard_examples
        self.output_head: nn.Linear | None = None  # Set by LEGOLLM

    @property
    def dim(self) -> int:
        """Model dimension (delegated to transformer)."""
        return self.transformer.dim

    @property
    def num_heads(self) -> int:
        """Number of attention heads (delegated to transformer)."""
        return self.transformer.num_heads

    @property
    def num_layers(self) -> int:
        """Number of layers (delegated to transformer)."""
        return self.transformer.num_layers

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through transformer with exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            h_out: Output hidden states (batch_size, seq_len, dim)
            logits: Output logits (batch_size, seq_len, vocab_size)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
            hidden_history: List of all hidden states for analysis
        """
        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")

        # Transformer forward with hidden history
        h_out, hidden_history = self.transformer(h)

        # Output logits
        logits = self.output_head(h_out)

        # Exit decision using exit_fn
        should_exit = self.exit_fn(hidden_history, self.threshold)

        return h_out, logits, should_exit, hidden_history

    def set_output_head(self, output_head: nn.Linear) -> None:
        """Set the shared output head reference."""
        self.output_head = output_head

    def collect_hard_examples(
        self,
        data: "SequenceData",
        hard_ratio: float,
        batch_size: int,
    ) -> "SequenceData":
        """
        Collect hard examples from data after training.

        Hard examples are tokens with low cos_sim (large change through the block).
        This method also sets self.threshold for early exit during inference.

        Args:
            data: SequenceData to collect hard examples from
            hard_ratio: Ratio of tokens to collect as hard examples (0.0-1.0)
            batch_size: Batch size for processing

        Returns:
            SequenceData with hard examples only (output hidden states)
        """
        from .sequence_data import SequenceData as SD

        device = next(self.parameters()).device
        self.eval()

        all_cos_sim: List[torch.Tensor] = []
        all_hidden_out: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
                h_out, _, _, hidden_history = self.forward(h)
                h_in = hidden_history[-2]
                cos_sim = compute_cos_sim(h_in, h_out)

                all_cos_sim.append(cos_sim.cpu())
                all_hidden_out.append(h_out.cpu())
                all_targets.append(y.cpu())

        cos_sim_all = torch.cat(all_cos_sim)
        hidden_out_all = torch.cat(all_hidden_out)
        targets_all = torch.cat(all_targets)

        # Compute threshold
        all_cos_flat = cos_sim_all.view(-1)
        if hard_ratio >= 1.0:
            self.threshold = float('inf')
        elif hard_ratio <= 0.0:
            self.threshold = float('-inf')
        else:
            self.threshold = float(torch.quantile(all_cos_flat, hard_ratio).item())

        # Token-level hard mask
        hard_token_mask = cos_sim_all < self.threshold
        hard_hidden = hidden_out_all[hard_token_mask]
        hard_targets = targets_all[hard_token_mask]

        num_hard_tokens = hard_hidden.shape[0]
        seq_len = data.seq_len
        dim = hidden_out_all.shape[-1]

        if num_hard_tokens == 0:
            return SD.empty(seq_len, dim, str(device))

        # Repack into sequences
        num_complete_sequences = num_hard_tokens // seq_len
        if num_complete_sequences == 0:
            return SD.empty(seq_len, dim, str(device))

        usable_tokens = num_complete_sequences * seq_len
        hard_hidden = hard_hidden[:usable_tokens].view(num_complete_sequences, seq_len, -1)
        hard_targets = hard_targets[:usable_tokens].view(num_complete_sequences, seq_len)

        return SD(hard_hidden, hard_targets)

    def transform_data(self, data: "SequenceData", batch_size: int) -> "SequenceData":
        """
        Transform SequenceData through this block.

        Passes all data through the block and returns output hidden states.
        Used to prepare validation data for the next block.

        Args:
            data: Input SequenceData
            batch_size: Batch size for processing

        Returns:
            SequenceData with transformed hidden states
        """
        from .sequence_data import SequenceData as SD

        device = next(self.parameters()).device
        self.eval()

        all_hidden: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
                h_out, _, _, _ = self.forward(h)
                all_hidden.append(h_out)
                all_targets.append(y)

        return SD(torch.cat(all_hidden), torch.cat(all_targets))
