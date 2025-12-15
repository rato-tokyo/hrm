"""
LEGO Framework - Block Training

Functions for training LEGOBlocks with hard example mining (token-level).
Uses hidden_history for exit decision and hard example collection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .block import LEGOBlock
    from .data import SequenceData

from .config import TrainerConfig
from .exit_fn import compute_cos_sim


def train_block(
    block: "LEGOBlock",
    train_data: "SequenceData",
    val_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> Tuple["SequenceData", Dict[str, Any]]:
    """
    Train a LEGOBlock and return hard examples for the next block.

    This function orchestrates the complete block training workflow:
    1. Train LM with early stopping (using val_data for validation)
    2. Compute cos_sim from hidden_history and set threshold
    3. Collect hard examples (token-level extraction, repacked into sequences)

    Args:
        block: LEGOBlock to train
        train_data: Training SequenceData (hidden_states, targets)
        val_data: Validation SequenceData for early stopping
        optimizer: Optimizer for block's parameters
        config: TrainerConfig with training hyperparameters

    Returns:
        Tuple of:
        - SequenceData: Hard examples for the next block (output hidden states)
        - Dict: Training statistics (train_ppls, val_ppls, best_epoch, etc.)
    """
    if block.output_head is None:
        raise RuntimeError("output_head not set. Call set_output_head() first.")

    if config.verbose:
        print(f"Training block: {len(train_data)} train, {len(val_data)} val sequences")
        print(f"  ({train_data.num_tokens} train, {val_data.num_tokens} val tokens)")

    # 1. Train LM with early stopping
    lm_stats = _train_lm(block, train_data, val_data, optimizer, config)

    # 2. Compute cos_sim and collect hard examples
    block.eval()
    hard_examples, threshold = _collect_hard_examples(block, train_data, config)
    block.threshold = threshold

    # 3. Build final statistics
    actual_hard_ratio = hard_examples.num_tokens / train_data.num_tokens if train_data.num_tokens > 0 else 0.0
    stats: Dict[str, Any] = {
        **lm_stats,
        'hard_ratio': actual_hard_ratio,
        'threshold': threshold,
    }

    if config.verbose:
        print(f"  Threshold (cos_sim): {threshold:.4f}")
        print(f"  Hard examples: {len(hard_examples)} sequences ({hard_examples.num_tokens} tokens, {actual_hard_ratio*100:.1f}%)")

    return hard_examples, stats


def _collect_hard_examples(
    block: "LEGOBlock",
    train_data: "SequenceData",
    config: TrainerConfig,
) -> Tuple["SequenceData", float]:
    """
    Collect hard examples using cos_sim from hidden_history.

    Low cos_sim = large change in layer = hard token = should NOT exit.

    Args:
        block: LEGOBlock (trained)
        train_data: Training SequenceData
        config: TrainerConfig with hard_ratio

    Returns:
        Tuple of:
        - SequenceData with hard examples only
        - threshold: cos_sim threshold for early exit
    """
    from .data import SequenceData

    device = next(block.parameters()).device
    all_cos_sim: List[torch.Tensor] = []
    all_hidden_out: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for h, y in train_data.to(str(device)).batches(config.batch_size, shuffle=False):
            # Get hidden_history from block
            h_out, _, _, hidden_history = block.forward(h)

            # Compute cos_sim from last two hidden states
            h_in = hidden_history[-2]
            cos_sim = compute_cos_sim(h_in, h_out)

            all_cos_sim.append(cos_sim.cpu())
            all_hidden_out.append(h_out.cpu())
            all_targets.append(y.cpu())

    cos_sim_all = torch.cat(all_cos_sim)  # (num_sequences, seq_len)
    hidden_out_all = torch.cat(all_hidden_out)  # (num_sequences, seq_len, dim)
    targets_all = torch.cat(all_targets)  # (num_sequences, seq_len)

    # Compute threshold: quantile such that hard_ratio tokens are hard
    # Hard tokens have LOW cos_sim (large change), so threshold = hard_ratio quantile
    all_cos_flat = cos_sim_all.view(-1)
    hard_ratio = config.hard_ratio

    if hard_ratio >= 1.0:
        threshold = float('inf')  # All tokens are hard
    elif hard_ratio <= 0.0:
        threshold = float('-inf')  # No tokens are hard
    else:
        # Threshold = hard_ratio quantile
        # Tokens with cos_sim < threshold are hard
        threshold = float(torch.quantile(all_cos_flat, hard_ratio).item())

    if config.verbose:
        print(f"  cos_sim stats: mean={all_cos_flat.mean():.4f}, std={all_cos_flat.std():.4f}")

    # Token-level hard mask: cos_sim < threshold (low similarity = hard)
    hard_token_mask = cos_sim_all < threshold  # (num_sequences, seq_len)

    # Extract only hard tokens
    hard_hidden = hidden_out_all[hard_token_mask]  # (num_hard_tokens, dim)
    hard_targets = targets_all[hard_token_mask]  # (num_hard_tokens,)

    num_hard_tokens = hard_hidden.shape[0]
    seq_len = train_data.seq_len
    dim = hidden_out_all.shape[-1]
    device_str = str(next(block.parameters()).device)

    if num_hard_tokens == 0:
        return SequenceData.empty(seq_len, dim, device_str), threshold

    # Repack into sequences
    num_complete_sequences = num_hard_tokens // seq_len
    if num_complete_sequences == 0:
        return SequenceData.empty(seq_len, dim, device_str), threshold

    usable_tokens = num_complete_sequences * seq_len
    hard_hidden = hard_hidden[:usable_tokens].view(num_complete_sequences, seq_len, -1)
    hard_targets = hard_targets[:usable_tokens].view(num_complete_sequences, seq_len)

    return SequenceData(hard_hidden, hard_targets), threshold


def _train_lm(
    block: "LEGOBlock",
    train_data: "SequenceData",
    val_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> Dict[str, Any]:
    """
    Train language model with early stopping.

    Args:
        block: LEGOBlock to train
        train_data: Training SequenceData
        val_data: Validation SequenceData
        optimizer: Optimizer for block's parameters
        config: TrainerConfig with training hyperparameters

    Returns:
        Dict with train_ppls, val_ppls, best_epoch, best_val_ppl, total_epochs, stopped_early
    """
    device = next(block.parameters()).device
    best_ppl = float('inf')
    best_state: Dict[str, torch.Tensor] | None = None
    patience_counter = 0
    best_epoch = 0
    epoch = 0

    train_ppls: List[float] = []
    val_ppls: List[float] = []

    for epoch in range(config.max_epochs):
        # Training epoch
        train_ppl = _train_epoch(block, train_data, optimizer, config)
        train_ppls.append(train_ppl)

        # Validation
        val_ppl = _evaluate_ppl(block, val_data, config.batch_size)
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

        if config.verbose:
            status = "best" if is_best else f"{patience_counter}/{config.patience}"
            print(f"  Epoch {epoch+1}/{config.max_epochs}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f} [{status}]")

        if patience_counter >= config.patience:
            if config.verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        block.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {
        'train_ppls': train_ppls,
        'val_ppls': val_ppls,
        'best_epoch': best_epoch,
        'best_val_ppl': best_ppl,
        'total_epochs': epoch + 1,
        'stopped_early': patience_counter >= config.patience,
    }


def _train_epoch(
    block: "LEGOBlock",
    train_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> float:
    """
    Run one training epoch.

    Args:
        block: LEGOBlock to train
        train_data: Training SequenceData
        optimizer: Optimizer for block's parameters
        config: TrainerConfig with training hyperparameters

    Returns:
        Training perplexity for this epoch
    """
    import numpy as np

    device = next(block.parameters()).device
    block.train()
    total_loss = 0.0
    total_tokens = 0

    for h, y in train_data.to(str(device)).batches(config.batch_size, shuffle=True):
        optimizer.zero_grad()
        _, logits, _, _ = block.forward(h)

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

    return float(np.exp(total_loss / total_tokens))


def _evaluate_ppl(
    block: "LEGOBlock",
    data: "SequenceData",
    batch_size: int,
) -> float:
    """
    Evaluate perplexity on a dataset.

    Args:
        block: LEGOBlock to evaluate
        data: SequenceData to evaluate on
        batch_size: Batch size for evaluation

    Returns:
        Perplexity
    """
    import numpy as np

    device = next(block.parameters()).device
    block.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
            _, logits, _, _ = block.forward(h)
            batch_size_actual, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += batch_size_actual * seq_len

    return float(np.exp(total_loss / total_tokens))
