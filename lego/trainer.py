"""
LEGO Framework - Block Training

Functions for training LEGOBlocks with hard example mining (sequence-based).
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
    2. Train with early stopping (sequence-based for proper Attention)
    3. Collect hard examples (sequences containing low confidence tokens)
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

    for epoch in range(config.max_epochs):
        # Training
        block.train()
        total_loss = 0.0
        total_tokens = 0

        for h, y in train_data.to(str(device)).batches(config.batch_size, shuffle=True):
            # h: (batch_size, seq_len, dim)
            # y: (batch_size, seq_len)
            optimizer.zero_grad()
            _, logits, _ = block.forward(h)
            # logits: (batch_size, seq_len, vocab_size)

            # Language modeling loss (flatten for cross_entropy)
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='sum'
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += loss.item()
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


def _collect_hard_examples(
    block: "LEGOBlock",
    data: "SequenceData",
    device: torch.device,
    hard_ratio: float
) -> Tuple["SequenceData", float]:
    """
    Collect hard examples (sequences with low confidence tokens) after training.

    Uses ratio-based selection: sequences containing tokens in the bottom X% confidence.
    Also computes the threshold based on the confidence distribution.

    Args:
        block: Trained LEGOBlock
        data: Input SequenceData
        device: Device to run on
        hard_ratio: Ratio of tokens to consider as hard (0.0-1.0)

    Returns:
        Tuple of:
        - SequenceData with hard examples (output hidden states and targets)
        - threshold: Confidence threshold for early exit (quantile-based)
    """
    from .data import SequenceData

    block.eval()

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

            # Compute confidence from softmax max for each token
            confidence = F.softmax(logits, dim=-1).max(dim=-1).values  # (batch_size, seq_len)

            all_hidden_out.append(h_out)
            all_targets.append(y)
            all_confidences.append(confidence)

    if not all_hidden_out:
        return SequenceData.empty(data.seq_len, block.dim, str(device)), 1.0

    # Concatenate all
    hidden_out_cat = torch.cat(all_hidden_out)  # (num_sequences, seq_len, dim)
    targets_cat = torch.cat(all_targets)  # (num_sequences, seq_len)
    confidences_cat = torch.cat(all_confidences)  # (num_sequences, seq_len)

    # Compute threshold: quantile such that (1 - hard_ratio) tokens exit
    # e.g., hard_ratio=0.5 means top 50% exit, so threshold = 50th percentile
    all_confidences_flat = confidences_cat.view(-1)
    threshold = float(torch.quantile(all_confidences_flat, 1.0 - hard_ratio).item())

    # Select sequences that contain at least one hard token (confidence < threshold)
    # A sequence is "hard" if any of its tokens has low confidence
    min_confidence_per_seq = confidences_cat.min(dim=1).values  # (num_sequences,)
    hard_sequence_mask = min_confidence_per_seq < threshold

    num_hard_sequences = hard_sequence_mask.sum().item()
    if num_hard_sequences == 0:
        return SequenceData.empty(data.seq_len, block.dim, str(device)), threshold

    return SequenceData(
        hidden_out_cat[hard_sequence_mask],
        targets_cat[hard_sequence_mask]
    ), threshold
