"""
LEGO Framework - LEGOBlock

A block of transformer layers with early exit capability at the final layer.
Each block owns its layers and handles forward pass through them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING

from .modules import TransformerBlock

if TYPE_CHECKING:
    from .data import TrainingData


class LEGOBlock(nn.Module):
    """
    A block of transformer layers with early exit capability.

    Each LEGOBlock:
    - Owns multiple TransformerBlock layers
    - Has a threshold for token-level early exit decision
    - Can compute confidence for routing decisions
    - Can be trained independently (for hard example training)

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers in this block
        threshold: Confidence threshold for early exit (1.0 = no early exit)
        output_head: Shared output projection (reference, not owned)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        threshold: float = 1.0,
        output_head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.threshold = threshold
        self.output_head = output_head  # Shared reference, not owned

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all layers with exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            h: Output hidden states (batch_size, seq_len, dim)
            logits: Output logits (batch_size, seq_len, vocab_size)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")

        for layer in self.layers:
            h = layer(h)

        logits = self.output_head(h)
        confidence = F.softmax(logits, dim=-1).max(dim=-1).values
        should_exit = confidence >= self.threshold

        return h, logits, should_exit

    def set_output_head(self, output_head: nn.Linear) -> None:
        """Set the shared output head reference."""
        self.output_head = output_head

    def train_block(
        self,
        data: "TrainingData",
        optimizer: torch.optim.Optimizer,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 3,
        grad_clip: float = 1.0,
        val_ratio: float = 0.2,
        verbose: bool = True
    ) -> Tuple["TrainingData", Dict[str, Any]]:
        """
        Train this block and return hard examples for the next block.

        This method encapsulates the complete block training workflow:
        1. Split input data into train/val
        2. Train with early stopping
        3. Collect hard examples (low confidence tokens)
        4. Return hard examples as TrainingData for the next block

        Args:
            data: TrainingData containing (hidden_states, targets)
            optimizer: Optimizer for this block's parameters
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs
            patience: Early stopping patience
            grad_clip: Gradient clipping value
            val_ratio: Ratio of data for validation (default: 0.2)
            verbose: Print training progress

        Returns:
            Tuple of:
            - TrainingData: Hard examples for the next block
            - Dict: Training statistics (train_ppls, val_ppls, best_epoch, etc.)
        """
        import numpy as np

        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")

        # Split data
        train_data, val_data = data.split(train_ratio=1.0 - val_ratio)

        if verbose:
            print(f"Training block: {len(train_data)} train, {len(val_data)} val tokens")

        # Training state
        best_ppl = float('inf')
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        best_epoch = 0

        train_ppls: List[float] = []
        val_ppls: List[float] = []

        device = next(self.parameters()).device

        for epoch in range(max_epochs):
            # Training
            self.train()
            total_loss = 0.0
            num_batches = 0

            for h, y in train_data.to(str(device)).batches(batch_size):
                optimizer.zero_grad()
                _, logits, _ = self.forward(h)
                logits = logits.squeeze(1)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            train_ppl = float(np.exp(total_loss / num_batches))
            train_ppls.append(train_ppl)

            # Validation
            self.eval()
            val_loss = 0.0
            val_tokens = 0

            with torch.no_grad():
                for h, y in val_data.to(str(device)).batches(batch_size, shuffle=False):
                    _, logits, _ = self.forward(h)
                    logits = logits.squeeze(1)
                    loss = F.cross_entropy(logits, y, reduction='sum')
                    val_loss += loss.item()
                    val_tokens += len(y)

            val_ppl = float(np.exp(val_loss / val_tokens))
            val_ppls.append(val_ppl)

            # Early stopping check
            is_best = val_ppl < best_ppl
            if is_best:
                best_ppl = val_ppl
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1

            if verbose:
                status = "best" if is_best else f"{patience_counter}/{patience}"
                print(f"  Epoch {epoch+1}/{max_epochs}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f} [{status}]")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Collect hard examples
        hard_examples = self._collect_hard_examples(data, device)

        hard_ratio = len(hard_examples) / len(data) if len(data) > 0 else 0.0
        stats: Dict[str, Any] = {
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
            'best_epoch': best_epoch,
            'best_val_ppl': best_ppl,
            'total_epochs': epoch + 1,
            'stopped_early': patience_counter >= patience,
            'hard_ratio': hard_ratio,
        }

        if verbose:
            print(f"  Hard examples: {len(hard_examples)} tokens ({hard_ratio*100:.1f}%)")

        return hard_examples, stats

    def _collect_hard_examples(
        self,
        data: "TrainingData",
        device: torch.device
    ) -> "TrainingData":
        """
        Collect hard examples (low confidence tokens) after training.

        Uses the block's threshold to identify hard tokens.
        Returns the output hidden states (after this block) for hard tokens.

        Args:
            data: Input TrainingData
            device: Device to run on

        Returns:
            TrainingData with hard examples (output hidden states and targets)
        """
        from .data import TrainingData

        self.eval()

        hard_hidden_states: List[torch.Tensor] = []
        hard_targets: List[torch.Tensor] = []

        with torch.no_grad():
            # Process in batches to avoid OOM
            for h, y in data.to(str(device)).batches(batch_size=256, shuffle=False):
                # Forward through this block
                h_out, _, should_exit = self.forward(h)
                should_exit = should_exit.squeeze(1)  # (batch_size,)

                # Hard tokens are those that should NOT exit (need more processing)
                hard_mask = ~should_exit

                if hard_mask.any():
                    # Get output hidden states for hard tokens
                    hard_hidden_states.append(h_out.squeeze(1)[hard_mask])
                    hard_targets.append(y[hard_mask])

        if not hard_hidden_states:
            # No hard examples found - return empty
            return TrainingData.empty(self.dim, str(device))

        return TrainingData(
            torch.cat(hard_hidden_states),
            torch.cat(hard_targets)
        )
