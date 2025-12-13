"""
LEGO Framework - Hard Example Mining Utilities

Two-phase training strategy that focuses computational resources on hard examples:
1. Phase 1: Train shallow model on all data
2. Phase 2: Extend model with additional block, train only on hard examples
3. Inference: Early exit routing using block thresholds

References:
- LEGO: Layered Ensemble with Gradual Optimization
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- Early Exit: BranchyNet (2016), Teerapittayanon et al. (2016)
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get available compute device (CUDA if available, otherwise CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def create_synthetic_data(
    num_batches: int = 4,
    batch_size: int = 8,
    seq_len: int = 16,
    vocab_size: int = 100
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic synthetic data for testing."""
    batches = []
    for i in range(num_batches):
        torch.manual_seed(42 + i)  # Deterministic per batch
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        batches.append((x, y))
    return batches


# ==============================================================================
# LEGO Utility Functions
# ==============================================================================

def compute_confidence_threshold(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    target_ratio: float,
    device: str,
    block_idx: int
) -> float:
    """
    Compute confidence threshold to achieve target hard example ratio.

    The threshold is set such that approximately target_ratio of examples
    fall below this confidence value (i.e., classified as hard examples).

    Args:
        model: Trained model to evaluate
        val_batches: Validation data batches
        target_ratio: Desired ratio of hard examples (e.g., 0.5 for 50%)
        device: Device to run computation on
        block_idx: Block index to compute confidence at (default: 0)

    Returns:
        Confidence threshold value
    """
    model.eval()
    all_confidences: List[torch.Tensor] = []

    with torch.no_grad():
        for x, _ in val_batches:
            x = x.to(device)
            # Get hidden states after specified block
            h = model.get_hidden_states(x, up_to_block=block_idx)
            _, confidence = model.blocks[block_idx].compute_confidence(h)
            all_confidences.append(confidence.view(-1))

    all_confidences_tensor = torch.cat(all_confidences)
    threshold = torch.quantile(all_confidences_tensor, target_ratio).item()

    return threshold


def collect_hard_examples(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: str,
    block_idx: int
) -> Dict[str, torch.Tensor]:
    """
    Collect hard examples (low confidence samples) from validation set.

    Hard examples are tokens where the model's prediction confidence falls
    below the threshold at the specified block. These are challenging for
    the block and will be passed to the next block for training.

    Args:
        model: Trained model
        val_batches: Validation data batches
        threshold: Confidence threshold for identifying hard examples
        device: Device to run computation on
        block_idx: Block index to compute confidence at (default: 0)

    Returns:
        Dictionary containing:
        - 'hidden_states': Hidden states after block (input for next block)
        - 'targets': Ground truth labels
    """
    model.eval()

    hard_hidden_states = []
    hard_targets = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            # Get hidden states after specified block
            h = model.get_hidden_states(x, up_to_block=block_idx)
            _, confidence = model.blocks[block_idx].compute_confidence(h)

            mask = confidence < threshold

            if mask.any():
                h_flat = h.view(-1, h.shape[-1])
                y_flat = y.view(-1)
                mask_flat = mask.view(-1)

                hard_hidden_states.append(h_flat[mask_flat])
                hard_targets.append(y_flat[mask_flat])

    if not hard_hidden_states:
        raise ValueError("No hard examples found. Try increasing threshold.")

    return {
        'hidden_states': torch.cat(hard_hidden_states),
        'targets': torch.cat(hard_targets),
    }


def split_hard_examples(
    hard_examples: Dict[str, torch.Tensor],
    train_ratio: float
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Split hard examples into train and validation sets.

    Args:
        hard_examples: Dictionary with 'hidden_states' and 'targets'
        train_ratio: Ratio of data to use for training (e.g., 0.8 for 80%)

    Returns:
        Tuple of (train_examples, val_examples) dictionaries
    """
    hidden_states = hard_examples['hidden_states']
    targets = hard_examples['targets']

    num_samples = len(targets)
    num_train = int(num_samples * train_ratio)

    # Shuffle indices
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_examples = {
        'hidden_states': hidden_states[train_indices],
        'targets': targets[train_indices],
    }
    val_examples = {
        'hidden_states': hidden_states[val_indices],
        'targets': targets[val_indices],
    }

    return train_examples, val_examples


def create_hard_example_loader(
    hard_examples: Dict[str, torch.Tensor],
    batch_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create batched dataloader from collected hard examples.

    Args:
        hard_examples: Dictionary with 'hidden_states' and 'targets'
        batch_size: Number of examples per batch

    Returns:
        List of batches (hidden_state, target) tuples
    """
    hidden_states = hard_examples['hidden_states']
    targets = hard_examples['targets']

    num_samples = len(targets)
    indices = torch.randperm(num_samples)

    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        # Add seq_len=1 dimension for compatibility with forward_from_block
        h_batch = hidden_states[batch_indices].unsqueeze(1)
        t_batch = targets[batch_indices]
        batches.append((h_batch, t_batch))

    return batches


