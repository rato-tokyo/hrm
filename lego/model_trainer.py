"""
LEGO Framework - LEGOLLM Training and Evaluation

Functions for training complete LEGOLLM models (all blocks) and evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import LEGOLLM
    from .block import LEGOBlock
    from .data import SequenceData

from .config import TrainerConfig
from .trainer import train_block
from .data import SequenceData


def train_legollm(
    model: "LEGOLLM",
    train_data: "SequenceData",
    val_data: "SequenceData",
    config: TrainerConfig,
    lr_decay: float,
) -> Dict[str, Any]:
    """
    Train all blocks in LEGOLLM sequentially.

    Training flow:
    1. Train Block 0 on all train data -> collect hard examples
    2. Train Block 1 on hard examples -> collect hard examples
    3. ... continue for all blocks

    Args:
        model: LEGOLLM model to train
        train_data: Training SequenceData (embedded tokens as hidden states)
        val_data: Validation SequenceData for early stopping
        config: TrainerConfig for training
        lr_decay: Learning rate decay factor for each subsequent block

    Returns:
        Dict with per-block statistics and overall training info
    """
    is_verbose = config.verbose

    all_stats: List[Dict[str, Any]] = []
    current_train_data = train_data
    current_val_data = val_data

    for block_idx, block in enumerate(model.blocks):
        is_last_block = (block_idx == len(model.blocks) - 1)

        if is_verbose:
            print(f"\n{'=' * 60}")
            print(f"Training Block {block_idx}")
            print("=" * 60)

        if len(current_train_data) == 0:
            if is_verbose:
                print(f"No data for Block {block_idx} - skipping")
            break

        # Decay learning rate for deeper blocks
        block_lr = config.lr * (lr_decay ** block_idx)
        block_config = TrainerConfig(
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            patience=config.patience,
            grad_clip=config.grad_clip,
            val_ratio=config.val_ratio,
            hard_ratio=config.hard_ratio,
            lr=block_lr,
            verbose=config.verbose,
        )

        optimizer = torch.optim.AdamW(block.parameters(), lr=block_lr)

        hard_data, stats = train_block(
            block=block,
            train_data=current_train_data,
            val_data=current_val_data,
            optimizer=optimizer,
            config=block_config,
        )

        all_stats.append({
            'block_idx': block_idx,
            'lr': block_lr,
            **stats,
        })

        if is_verbose:
            print(f"\nBlock {block_idx} Results:")
            print(f"  Best PPL: {stats['best_val_ppl']:.2f}")
            print(f"  Threshold: {stats['threshold']:.4f}")
            print(f"  Hard examples: {len(hard_data)} sequences ({hard_data.num_tokens} tokens)")

        # Update data for next block (except for last block)
        if not is_last_block:
            current_train_data = hard_data
            # Transform val_data through the trained block
            current_val_data = _transform_data_through_block(block, current_val_data, config.batch_size)

    return {
        'block_stats': all_stats,
        'num_blocks_trained': len(all_stats),
    }


def _transform_data_through_block(
    block: "LEGOBlock",
    data: "SequenceData",
    batch_size: int,
) -> "SequenceData":
    """
    Transform SequenceData through a trained block.

    Args:
        block: Trained LEGOBlock
        data: Input SequenceData
        batch_size: Batch size for processing

    Returns:
        SequenceData with transformed hidden states
    """
    device = next(block.parameters()).device
    block.eval()

    all_hidden: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
            h_out, _, _ = block.forward(h)
            all_hidden.append(h_out)
            all_targets.append(y)

    return SequenceData(
        torch.cat(all_hidden),
        torch.cat(all_targets),
    )


def evaluate_legollm(
    model: "LEGOLLM",
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Evaluate LEGOLLM with TRUE Early Exit.

    Args:
        model: Trained LEGOLLM model
        val_batches: List of (x, y) batches for validation

    Returns:
        Dict with ppl, accuracy, shallow_ratio, compute_cost, exit_counts
    """
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct = 0
    all_exit_counts = [0] * len(model.blocks)

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            logits, stats = model.forward(x, return_stats=True)

            # Accumulate exit counts
            for i, count in enumerate(stats['exit_counts']):
                all_exit_counts[i] += count

            # Loss and accuracy
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            total_loss += F.cross_entropy(logits_flat, y_flat, reduction='sum').item()
            total_tokens += y_flat.numel()
            correct += int((logits_flat.argmax(dim=-1) == y_flat).sum().item())

    ppl = float(np.exp(total_loss / total_tokens))
    acc = correct / total_tokens

    # Compute shallow ratio
    shallow_exits = sum(all_exit_counts[:-1])
    shallow_ratio = shallow_exits / total_tokens if total_tokens > 0 else 0.0

    # Compute cost
    total_layers_computed = 0
    layers_so_far = 0
    for block_idx, count in enumerate(all_exit_counts):
        layers_so_far += model.blocks[block_idx].num_layers
        total_layers_computed += count * layers_so_far
    compute_cost = total_layers_computed / (total_tokens * model.num_layers) if total_tokens > 0 else 1.0

    return {
        'ppl': ppl,
        'accuracy': acc,
        'shallow_ratio': shallow_ratio,
        'compute_cost': compute_cost,
        'compute_savings': 1.0 - compute_cost,
        'exit_counts': all_exit_counts,
        'total_tokens': total_tokens,
    }


def create_sequence_data(
    model: "LEGOLLM",
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> "SequenceData":
    """
    Create SequenceData by embedding tokens.

    Args:
        model: LEGOLLM model (for embedding layer)
        batches: List of (x, y) batches

    Returns:
        SequenceData with embedded hidden states and targets
    """
    device = next(model.parameters()).device

    all_hidden: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            h = model.embedding(x)
            all_hidden.append(h)
            all_targets.append(y)

    return SequenceData(
        torch.cat(all_hidden),
        torch.cat(all_targets),
    )
