"""
LEGO Framework - Exit Classifier Training

Functions for training exit classifiers and collecting hard examples.
Exit classifier predicts per-token loss directly (BDR-style approach).

These functions are decoupled from LEGOBlock internals - they only need
the exit_classifier, hidden_states, and precomputed loss values.
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
    exit_labels: torch.Tensor,
    lr: float,
    num_epochs: int,
    is_verbose: bool,
) -> None:
    """
    Train exit_classifier to predict per-token loss.

    BDR-style approach: directly predict loss values, no sigmoid.
    Lower predicted loss = easier token = should exit early.

    Args:
        exit_classifier: ExitClassifier to train
        hidden_states: Hidden states (batch, seq_len, dim)
        exit_labels: Per-token loss values (batch, seq_len)
        lr: Learning rate
        num_epochs: Number of training epochs
        is_verbose: Print progress
    """
    if is_verbose:
        print("  Training exit_classifier (BDR-style: predict loss)...")

    # Flatten to (num_tokens, dim) and (num_tokens,)
    if hidden_states.dim() == 3:
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, dim)
        exit_labels = exit_labels.view(-1)

    num_tokens = hidden_states.shape[0]

    # Setup optimizer
    exit_optimizer = torch.optim.Adam(exit_classifier.parameters(), lr=lr)

    # Training loop
    exit_classifier.train()
    for epoch in range(num_epochs):
        exit_optimizer.zero_grad()

        # Forward pass - reshape for exit_classifier
        h_reshaped = hidden_states.unsqueeze(1)  # (num_tokens, 1, dim)
        exit_preds = exit_classifier.compute_confidence(h_reshaped).squeeze(1)  # (num_tokens,)

        loss = F.mse_loss(exit_preds, exit_labels, reduction='sum')

        loss.backward()
        exit_optimizer.step()

        avg_loss = loss.item() / num_tokens
        avg_label = exit_labels.mean().item()
        avg_pred = exit_preds.mean().item()
        if is_verbose:
            print(f"    Exit epoch {epoch+1}/{num_epochs}: mse={avg_loss:.4f}, "
                  f"avg_label={avg_label:.2f}, avg_pred={avg_pred:.2f}")

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
    Collect hard examples (token-level) based on predicted loss.

    BDR-style: uses predicted loss for selection.
    High predicted loss = hard token = should NOT exit.

    Uses ratio-based selection: extracts tokens with predicted loss in the top X%.
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
        - threshold: Predicted loss threshold for early exit (quantile-based)
    """
    from .data import SequenceData

    exit_classifier.eval()
    device = hidden_states.device
    dim = hidden_states.shape[-1]

    # Compute predicted loss for all tokens
    with torch.no_grad():
        predicted_loss = exit_classifier.compute_confidence(hidden_states)  # (num_sequences, seq_len)

    # Compute threshold: quantile such that hard_ratio tokens are collected as hard
    # Hard tokens have HIGH predicted loss, so we use (1 - hard_ratio) quantile
    all_preds_flat = predicted_loss.view(-1)
    if hard_ratio >= 1.0:
        threshold = float('-inf')  # All tokens are hard
    elif hard_ratio <= 0.0:
        threshold = float('inf')  # No tokens are hard
    else:
        # Threshold = (1 - hard_ratio) quantile
        # Tokens with predicted_loss > threshold are hard
        threshold = float(torch.quantile(all_preds_flat, 1.0 - hard_ratio).item())

    # Token-level hard mask: predicted_loss > threshold (high loss = hard)
    hard_token_mask = predicted_loss > threshold  # (num_sequences, seq_len)

    # Ensure targets is on same device as hidden_states
    targets = targets.to(device)

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
