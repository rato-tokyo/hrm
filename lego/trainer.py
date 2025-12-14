"""
LEGO Framework - Block Training

Functions for training LEGOBlocks with hard example mining (token-level).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .block import LEGOBlock
    from .data import SequenceData

from .config import TrainerConfig


def train_block(
    block: "LEGOBlock",
    data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> Tuple["SequenceData", Dict[str, Any]]:
    """
    Train a LEGOBlock and return hard examples for the next block.

    This function encapsulates the complete block training workflow:
    1. Split input data into train/val
    2. Train with early stopping
    3. Collect hard examples (token-level extraction, repacked into sequences)
    4. Set block's threshold based on confidence distribution
    5. Return hard examples as SequenceData for the next block

    Args:
        block: LEGOBlock to train
        data: SequenceData containing (hidden_states, targets) as sequences
        optimizer: Optimizer for block's parameters
        config: TrainerConfig with training hyperparameters

    Returns:
        Tuple of:
        - SequenceData: Hard examples for the next block (output hidden states)
        - Dict: Training statistics (train_ppls, val_ppls, best_epoch, etc.)
    """
    import numpy as np

    is_verbose = config.verbose

    if block.output_head is None:
        raise RuntimeError("output_head not set. Call set_output_head() first.")

    # Split data
    train_data, val_data = data.split(train_ratio=1.0 - config.val_ratio)

    if is_verbose:
        print(f"Training block: {len(train_data)} train, {len(val_data)} val sequences")
        print(f"  ({train_data.num_tokens} train, {val_data.num_tokens} val tokens)")

    # Training state
    best_ppl = float('inf')
    best_state: Dict[str, torch.Tensor] | None = None
    patience_counter = 0
    best_epoch = 0
    epoch = 0

    train_ppls: List[float] = []
    val_ppls: List[float] = []

    device = next(block.parameters()).device

    is_joint_mode = config.exit_classifier_mode == "joint"

    for epoch in range(config.max_epochs):
        # Training
        block.train()
        total_loss = 0.0
        total_exit_loss = 0.0
        total_tokens = 0

        for h, y in train_data.to(str(device)).batches(config.batch_size, shuffle=True):
            # h: (batch_size, seq_len, dim)
            # y: (batch_size, seq_len)
            optimizer.zero_grad()
            h_out, logits, _ = block.forward(h)
            # logits: (batch_size, seq_len, vocab_size)

            # Language modeling loss (flatten for cross_entropy)
            batch_size, seq_len, vocab_size = logits.shape
            lm_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='sum'
            )

            loss = lm_loss

            # Joint mode: add exit_classifier BCE loss
            if is_joint_mode:
                # Label: 1 if prediction is correct, 0 otherwise
                preds = logits.argmax(dim=-1)  # (batch_size, seq_len)
                exit_labels = (preds == y).float()  # (batch_size, seq_len)

                # exit_classifier output
                exit_logits = block.exit_classifier(h_out).squeeze(-1)  # (batch_size, seq_len)
                exit_loss = F.binary_cross_entropy_with_logits(
                    exit_logits.view(-1),
                    exit_labels.view(-1),
                    reduction='sum'
                )
                loss = lm_loss + exit_loss
                total_exit_loss += exit_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += lm_loss.item()
            total_tokens += batch_size * seq_len

        train_ppl = float(np.exp(total_loss / total_tokens))
        train_ppls.append(train_ppl)

        # Validation
        block.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for h, y in val_data.to(str(device)).batches(config.batch_size, shuffle=False):
                _, logits, _ = block.forward(h)
                batch_size, seq_len, vocab_size = logits.shape
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                    reduction='sum'
                )
                val_loss += loss.item()
                val_tokens += batch_size * seq_len

        val_ppl = float(np.exp(val_loss / val_tokens))
        val_ppls.append(val_ppl)

        # Early stopping check
        is_best = val_ppl < best_ppl
        if is_best:
            best_ppl = val_ppl
            best_state = {k: v.cpu().clone() for k, v in block.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if is_verbose:
            status = "best" if is_best else f"{patience_counter}/{config.patience}"
            print(f"  Epoch {epoch+1}/{config.max_epochs}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f} [{status}]")

        if patience_counter >= config.patience:
            if is_verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        block.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Post mode: train exit_classifier separately after LM training
    if config.exit_classifier_mode == "post":
        _train_exit_classifier(block, data, device, config, is_verbose)

    # Collect hard examples and set threshold based on confidence distribution
    hard_examples, threshold = _collect_hard_examples(block, data, device, config.hard_ratio)
    block.threshold = threshold

    actual_hard_ratio = hard_examples.num_tokens / data.num_tokens if data.num_tokens > 0 else 0.0
    stats: Dict[str, Any] = {
        'train_ppls': train_ppls,
        'val_ppls': val_ppls,
        'best_epoch': best_epoch,
        'best_val_ppl': best_ppl,
        'total_epochs': epoch + 1,
        'stopped_early': patience_counter >= config.patience,
        'hard_ratio': actual_hard_ratio,
        'threshold': threshold,
    }

    if is_verbose:
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Hard examples: {len(hard_examples)} sequences ({hard_examples.num_tokens} tokens, {actual_hard_ratio*100:.1f}%)")

    return hard_examples, stats


def _train_exit_classifier(
    block: "LEGOBlock",
    data: "SequenceData",
    device: torch.device,
    config: TrainerConfig,
    is_verbose: bool
) -> None:
    """
    Train exit_classifier separately after LM training (post mode).

    Freezes transformer and output_head, trains only exit_classifier.
    Label: 1 if prediction is correct, 0 otherwise.

    Args:
        block: LEGOBlock with trained transformer
        data: Full training data
        device: Device to run on
        config: TrainerConfig
        is_verbose: Print progress
    """
    if is_verbose:
        print("  Training exit_classifier (post mode)...")

    # Freeze all except exit_classifier
    for param in block.transformer.parameters():
        param.requires_grad = False
    if block.output_head is not None:
        for param in block.output_head.parameters():
            param.requires_grad = False

    # Only exit_classifier is trainable
    exit_optimizer = torch.optim.Adam(block.exit_classifier.parameters(), lr=config.lr)

    # Quick training loop (fewer epochs, no early stopping)
    num_epochs = min(10, config.max_epochs)

    for epoch in range(num_epochs):
        block.train()
        total_loss = 0.0
        total_tokens = 0
        correct_preds = 0

        for h, y in data.to(str(device)).batches(config.batch_size, shuffle=True):
            exit_optimizer.zero_grad()

            with torch.no_grad():
                h_out = block.transformer(h)
                logits = block.output_head(h_out)
                preds = logits.argmax(dim=-1)
                exit_labels = (preds == y).float()

            # Train exit_classifier
            exit_logits = block.exit_classifier(h_out).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                exit_logits.view(-1),
                exit_labels.view(-1),
                reduction='sum'
            )

            loss.backward()
            exit_optimizer.step()

            total_loss += loss.item()
            batch_size, seq_len = y.shape
            total_tokens += batch_size * seq_len
            correct_preds += exit_labels.sum().item()

        avg_loss = total_loss / total_tokens
        accuracy = correct_preds / total_tokens
        if is_verbose:
            print(f"    Exit epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, acc={accuracy*100:.1f}%")

    # Unfreeze all parameters
    for param in block.transformer.parameters():
        param.requires_grad = True
    if block.output_head is not None:
        for param in block.output_head.parameters():
            param.requires_grad = True

    if is_verbose:
        print("  Exit_classifier training complete")


def _collect_hard_examples(
    block: "LEGOBlock",
    data: "SequenceData",
    device: torch.device,
    hard_ratio: float
) -> Tuple["SequenceData", float]:
    """
    Collect hard examples (token-level) after training.

    Uses ratio-based selection: extracts only tokens with confidence in the bottom X%.
    Hard tokens are then repacked into new sequences of the same seq_len.

    Args:
        block: Trained LEGOBlock
        data: Input SequenceData
        device: Device to run on
        hard_ratio: Ratio of tokens to consider as hard (0.0-1.0)

    Returns:
        Tuple of:
        - SequenceData with hard examples only (output hidden states and targets)
        - threshold: Confidence threshold for early exit (quantile-based)
    """
    from .data import SequenceData

    block.eval()
    seq_len = data.seq_len

    all_hidden_out: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_confidences: List[torch.Tensor] = []

    with torch.no_grad():
        # Process sequences
        for h, y in data.to(str(device)).batches(batch_size=32, shuffle=False):
            # h: (batch_size, seq_len, dim)
            # y: (batch_size, seq_len)
            h_out, logits, _ = block.forward(h)
            # h_out: (batch_size, seq_len, dim)
            # logits: (batch_size, seq_len, vocab_size)

            # Compute confidence using exit_classifier (not softmax)
            confidence = torch.sigmoid(block.exit_classifier(h_out)).squeeze(-1)  # (batch_size, seq_len)

            all_hidden_out.append(h_out)
            all_targets.append(y)
            all_confidences.append(confidence)

    if not all_hidden_out:
        return SequenceData.empty(seq_len, block.dim, str(device)), 1.0

    # Concatenate all
    hidden_out_cat = torch.cat(all_hidden_out)  # (num_sequences, seq_len, dim)
    targets_cat = torch.cat(all_targets)  # (num_sequences, seq_len)
    confidences_cat = torch.cat(all_confidences)  # (num_sequences, seq_len)

    # Compute threshold: quantile such that hard_ratio tokens are collected as hard
    # e.g., hard_ratio=0.5 means bottom 50% are hard, so threshold = 50th percentile
    # confidence < threshold → hard token
    all_confidences_flat = confidences_cat.view(-1)
    if hard_ratio >= 1.0:
        threshold = float('inf')  # All tokens are hard
    else:
        threshold = float(torch.quantile(all_confidences_flat, hard_ratio).item())

    # Token-level hard mask: confidence < threshold
    hard_token_mask = confidences_cat < threshold  # (num_sequences, seq_len)

    # Extract only hard tokens
    hard_hidden = hidden_out_cat[hard_token_mask]  # (num_hard_tokens, dim)
    hard_targets = targets_cat[hard_token_mask]  # (num_hard_tokens,)

    num_hard_tokens = hard_hidden.shape[0]
    if num_hard_tokens == 0:
        return SequenceData.empty(seq_len, block.dim, str(device)), threshold

    # Repack into sequences (truncate remainder that doesn't fill a complete sequence)
    num_complete_sequences = num_hard_tokens // seq_len
    if num_complete_sequences == 0:
        return SequenceData.empty(seq_len, block.dim, str(device)), threshold

    usable_tokens = num_complete_sequences * seq_len
    hard_hidden = hard_hidden[:usable_tokens].view(num_complete_sequences, seq_len, -1)
    hard_targets = hard_targets[:usable_tokens].view(num_complete_sequences, seq_len)

    return SequenceData(hard_hidden, hard_targets), threshold
