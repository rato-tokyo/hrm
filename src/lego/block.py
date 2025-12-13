"""
LEGO Framework - LEGOBlock

TransformerBlock wrapper with early exit capability.
Separates standard transformer functionality from LEGO-specific features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .data import TrainingData

from .modules import TransformerBlock


class LEGOBlock(nn.Module):
    """
    TransformerBlock with early exit capability.

    Wraps a standard TransformerBlock and adds:
    - Lightweight exit_classifier for confidence prediction
    - Threshold for token-level early exit decision (set automatically by fit())

    This separation allows:
    - Standard TransformerBlock to be used independently
    - Easy replacement of transformer implementation (e.g., Flash Attention)
    - Clear distinction between standard and LEGO-specific functionality

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers in this block
        output_head: Shared output projection (reference, not owned)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        output_head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.threshold = 1.0  # Set automatically by fit()
        self.output_head = output_head  # Shared reference, not owned

        # Standard transformer block (composition)
        self.transformer = TransformerBlock(dim, num_heads, num_layers)

        # LEGO-specific: Lightweight exit classifier (dim -> 1)
        self.exit_classifier = nn.Linear(dim, 1)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer with exit decision.

        Args:
            h: Hidden states (batch_size, seq_len, dim)

        Returns:
            h: Output hidden states (batch_size, seq_len, dim)
            logits: Output logits (batch_size, seq_len, vocab_size)
            should_exit: Boolean mask where True = should exit (batch_size, seq_len)
        """
        if self.output_head is None:
            raise RuntimeError("output_head not set. Call set_output_head() first.")

        # Standard transformer forward
        h = self.transformer(h)

        # Output logits
        logits = self.output_head(h)

        # LEGO-specific: Lightweight confidence from exit_classifier
        confidence = torch.sigmoid(self.exit_classifier(h)).squeeze(-1)
        should_exit = confidence >= self.threshold

        return h, logits, should_exit

    def set_output_head(self, output_head: nn.Linear) -> None:
        """Set the shared output head reference."""
        self.output_head = output_head

    def fit(
        self,
        data: "TrainingData",
        optimizer: torch.optim.Optimizer,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 3,
        grad_clip: float = 1.0,
        val_ratio: float = 0.2,
        hard_ratio: float = 0.5,
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
            hard_ratio: Ratio of tokens to collect as hard examples (default: 0.5)
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
                h_out, logits, _ = self.forward(h)
                logits = logits.squeeze(1)

                # Language modeling loss
                lm_loss = F.cross_entropy(logits, y)

                # Exit classifier loss: predict if token is correct
                with torch.no_grad():
                    predicted = logits.argmax(dim=-1)
                    is_correct = (predicted == y).float()

                exit_logits = self.exit_classifier(h_out).squeeze(-1).squeeze(-1)
                exit_loss = F.binary_cross_entropy_with_logits(exit_logits, is_correct)

                # Combined loss
                loss = lm_loss + exit_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                total_loss += lm_loss.item()
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

        # Collect hard examples and set threshold based on confidence distribution
        hard_examples, threshold = self._collect_hard_examples(data, device, hard_ratio)
        self.threshold = threshold

        actual_hard_ratio = len(hard_examples) / len(data) if len(data) > 0 else 0.0
        stats: Dict[str, Any] = {
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
            'best_epoch': best_epoch,
            'best_val_ppl': best_ppl,
            'total_epochs': epoch + 1,
            'stopped_early': patience_counter >= patience,
            'hard_ratio': actual_hard_ratio,
            'threshold': threshold,
        }

        if verbose:
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Hard examples: {len(hard_examples)} tokens ({actual_hard_ratio*100:.1f}%)")

        return hard_examples, stats

    def _collect_hard_examples(
        self,
        data: "TrainingData",
        device: torch.device,
        hard_ratio: float
    ) -> Tuple["TrainingData", float]:
        """
        Collect hard examples (low confidence tokens) after training.

        Uses ratio-based selection: collects the bottom X% tokens by confidence.
        Also computes the threshold based on the confidence distribution.

        Args:
            data: Input TrainingData
            device: Device to run on
            hard_ratio: Ratio of tokens to collect (0.0-1.0)

        Returns:
            Tuple of:
            - TrainingData with hard examples (output hidden states and targets)
            - threshold: Confidence threshold for early exit (quantile-based)
        """
        from .data import TrainingData

        self.eval()

        all_hidden_out: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_confidences: List[torch.Tensor] = []

        with torch.no_grad():
            # Process in batches to avoid OOM
            for h, y in data.to(str(device)).batches(batch_size=256, shuffle=False):
                # Forward through this block
                h_out, _, _ = self.forward(h)
                h_out = h_out.squeeze(1)  # (batch_size, dim)

                # Compute confidence from exit_classifier
                confidence = torch.sigmoid(self.exit_classifier(h_out)).squeeze(-1)  # (batch_size,)

                all_hidden_out.append(h_out)
                all_targets.append(y)
                all_confidences.append(confidence)

        if not all_hidden_out:
            return TrainingData.empty(self.dim, str(device)), 1.0

        # Concatenate all
        hidden_out_cat = torch.cat(all_hidden_out)  # (num_tokens, dim)
        targets_cat = torch.cat(all_targets)  # (num_tokens,)
        confidences_cat = torch.cat(all_confidences)  # (num_tokens,)

        # Compute threshold: quantile such that (1 - hard_ratio) tokens exit
        # e.g., hard_ratio=0.5 means top 50% exit, so threshold = 50th percentile
        threshold = float(torch.quantile(confidences_cat, 1.0 - hard_ratio).item())

        # Select bottom X% by confidence (ratio-based)
        num_hard = int(len(confidences_cat) * hard_ratio)
        if num_hard == 0:
            return TrainingData.empty(self.dim, str(device)), threshold

        # Get indices of tokens with lowest confidence
        _, hard_indices = torch.topk(confidences_cat, num_hard, largest=False)

        return TrainingData(
            hidden_out_cat[hard_indices],
            targets_cat[hard_indices]
        ), threshold
