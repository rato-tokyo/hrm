"""
LEGO Framework - Exit Classifier Training

Functions for training exit classifiers and collecting hard examples.
Exit classifier uses loss-based labels: exp(-cross_entropy_loss).

These functions are decoupled from LEGOBlock internals - they only need
the exit_classifier, hidden_states, logits, and targets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .exit_classifier import ExitClassifier
    from .data import SequenceData


def train_exit_classifier(
    exit_classifier: "ExitClassifier",
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    lr: float,
    num_epochs: int,
    is_verbose: bool,
) -> None:
    """
    Train exit_classifier using loss-based labels.

    Labels are computed as exp(-cross_entropy_loss) for each token.
    Higher confidence = lower loss = easier token.

    Args:
        exit_classifier: ExitClassifier to train
        hidden_states: Hidden states from block.forward() (num_tokens, dim) or (batch, seq_len, dim)
        logits: Logits from block.forward() (num_tokens, vocab_size) or (batch, seq_len, vocab_size)
        targets: Target token IDs (num_tokens,) or (batch, seq_len)
        lr: Learning rate
        num_epochs: Number of training epochs
        is_verbose: Print progress
    """
    if is_verbose:
        print("  Training exit_classifier...")

    # Flatten if needed
    if hidden_states.dim() == 3:
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, dim)
        logits = logits.view(-1, logits.shape[-1])
        targets = targets.view(-1)

    num_tokens = hidden_states.shape[0]

    # Compute exit labels from logits (detached, no gradient)
    with torch.no_grad():
        per_token_loss = F.cross_entropy(logits, targets, reduction='none')
        exit_labels = torch.exp(-per_token_loss)

    # Setup optimizer
    exit_optimizer = torch.optim.Adam(exit_classifier.parameters(), lr=lr)

    # Training loop
    exit_classifier.train()
    for epoch in range(num_epochs):
        exit_optimizer.zero_grad()

        # Forward pass - need to reshape for exit_classifier which expects (batch, seq_len, dim)
        # Use (num_tokens, 1, dim) to process all tokens
        h_reshaped = hidden_states.unsqueeze(1)  # (num_tokens, 1, dim)
        exit_preds = exit_classifier.compute_confidence(h_reshaped).squeeze(1)  # (num_tokens,)

        loss = F.mse_loss(exit_preds, exit_labels, reduction='sum')

        loss.backward()
        exit_optimizer.step()

        avg_loss = loss.item() / num_tokens
        avg_label = exit_labels.mean().item()
        if is_verbose:
            print(f"    Exit epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, avg_label={avg_label:.3f}")

    exit_classifier.eval()

    if is_verbose:
        print("  Exit_classifier training complete")


def collect_hard_examples(
    exit_classifier: "ExitClassifier",
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    seq_len: int,
    hard_ratio: float,
) -> Tuple["SequenceData", float]:
    """
    Collect hard examples (token-level) based on exit_classifier confidence.

    Uses ratio-based selection: extracts only tokens with confidence in the bottom X%.
    Hard tokens are then repacked into new sequences of the specified seq_len.

    Args:
        exit_classifier: Trained ExitClassifier
        hidden_states: Hidden states (num_sequences, seq_len, dim)
        targets: Target token IDs (num_sequences, seq_len)
        seq_len: Sequence length for repacking
        hard_ratio: Ratio of tokens to consider as hard (0.0-1.0)

    Returns:
        Tuple of:
        - SequenceData with hard examples only (output hidden states and targets)
        - threshold: Confidence threshold for early exit (quantile-based)
    """
    from .data import SequenceData

    exit_classifier.eval()
    device = hidden_states.device
    dim = hidden_states.shape[-1]

    # Compute confidence for all tokens
    with torch.no_grad():
        confidences = exit_classifier.compute_confidence(hidden_states)  # (num_sequences, seq_len)

    # Compute threshold: quantile such that hard_ratio tokens are collected as hard
    # e.g., hard_ratio=0.5 means bottom 50% are hard, so threshold = 50th percentile
    # confidence < threshold -> hard token
    all_confidences_flat = confidences.view(-1)
    if hard_ratio >= 1.0:
        threshold = float('inf')  # All tokens are hard
    else:
        threshold = float(torch.quantile(all_confidences_flat, hard_ratio).item())

    # Token-level hard mask: confidence < threshold
    hard_token_mask = confidences < threshold  # (num_sequences, seq_len)

    # Extract only hard tokens
    hard_hidden = hidden_states[hard_token_mask]  # (num_hard_tokens, dim)
    hard_targets = targets[hard_token_mask]  # (num_hard_tokens,)

    num_hard_tokens = hard_hidden.shape[0]
    if num_hard_tokens == 0:
        return SequenceData.empty(seq_len, dim, str(device)), threshold

    # Repack into sequences (truncate remainder that doesn't fill a complete sequence)
    num_complete_sequences = num_hard_tokens // seq_len
    if num_complete_sequences == 0:
        return SequenceData.empty(seq_len, dim, str(device)), threshold

    usable_tokens = num_complete_sequences * seq_len
    hard_hidden = hard_hidden[:usable_tokens].view(num_complete_sequences, seq_len, -1)
    hard_targets = hard_targets[:usable_tokens].view(num_complete_sequences, seq_len)

    return SequenceData(hard_hidden, hard_targets), threshold
