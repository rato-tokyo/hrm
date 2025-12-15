"""
LEGO Framework - LEGOLLM Training

Functions for training complete LEGOLLM models (all blocks).
"""

from __future__ import annotations

import torch
from typing import Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .legollm import LEGOLLM
    from .sequence_data import SequenceData

from .config import TrainerConfig
from .block_trainer import train_block
from .sequence_data import SequenceData


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
    all_stats: List[Dict[str, Any]] = []
    current_train_data = train_data
    current_val_data = val_data

    for block_idx, block in enumerate(model.blocks):
        is_last_block = (block_idx == len(model.blocks) - 1)

        if config.verbose:
            print(f"\n{'=' * 60}")
            print(f"Training Block {block_idx}")
            print("=" * 60)

        if len(current_train_data) == 0:
            if config.verbose:
                print(f"No data for Block {block_idx} - skipping")
            break

        # Decay learning rate for deeper blocks
        block_lr = config.lr * (lr_decay ** block_idx)
        block_config = TrainerConfig(
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            patience=config.patience,
            grad_clip=config.grad_clip,
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

        if config.verbose:
            print(f"\nBlock {block_idx} Results:")
            print(f"  Best PPL: {stats['best_val_ppl']:.2f}")
            print(f"  Threshold: {stats['threshold']:.4f}")
            print(f"  Hard examples: {len(hard_data)} sequences ({hard_data.num_tokens} tokens)")

        # Update data for next block (except for last block)
        if not is_last_block:
            current_train_data = hard_data
            # Transform val_data through the trained block
            current_val_data = block.transform_data(current_val_data, config.batch_size)

    return {
        'block_stats': all_stats,
        'num_blocks_trained': len(all_stats),
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
