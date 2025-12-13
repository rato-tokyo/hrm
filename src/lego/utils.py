"""
LEGO Framework - Hard Example Mining Utilities

Two-phase training strategy that focuses computational resources on hard examples:
1. Phase 1: Train shallow model on all data
2. Phase 2: Extend model with additional layers, train only on hard examples
3. Inference: Two-stage routing using Early Exit mechanism

References:
- LEGO: Layered Ensemble with Gradual Optimization
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- Early Exit: BranchyNet (2016), Teerapittayanon et al. (2016)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_routing_cost(
    shallow_count: int,
    deep_count: int,
    exit_layer: int,
    num_layers: int
) -> float:
    """
    Compute routing cost as fraction of full model computation.

    Cost = weighted average of layers computed per token, normalized by total layers.
    - Shallow tokens compute exit_layer layers
    - Deep tokens compute num_layers layers

    Args:
        shallow_count: Number of tokens using shallow path
        deep_count: Number of tokens using deep path
        exit_layer: Number of layers for shallow path
        num_layers: Total number of layers in model

    Returns:
        Compute cost as fraction (0.0 to 1.0)
    """
    total_count = shallow_count + deep_count
    if total_count == 0:
        return 1.0
    actual_layers = shallow_count * exit_layer + deep_count * num_layers
    return actual_layers / (total_count * num_layers)


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
    device: str
) -> float:
    """
    Compute confidence threshold to achieve target hard example ratio.

    The threshold is set such that approximately target_ratio of examples
    fall below this confidence value (i.e., classified as hard examples).

    Strategy:
    1. Compute confidence for all validation examples
    2. Find the target_ratio percentile of these confidences
    3. Use this percentile as the threshold

    Args:
        model: Trained model to evaluate
        val_batches: Validation data batches
        target_ratio: Desired ratio of hard examples (e.g., 0.5 for 50%)
        device: Device to run computation on

    Returns:
        Confidence threshold value
    """
    model.eval()
    all_confidences: List[torch.Tensor] = []

    with torch.no_grad():
        for x, _ in val_batches:
            x = x.to(device)

            # Get hidden states using model method
            h = model.get_hidden_states(x)

            # Compute confidence
            _, confidence = model.compute_confidence(h)
            all_confidences.append(confidence.view(-1))

    # Concatenate all confidences and compute threshold
    all_confidences_tensor = torch.cat(all_confidences)
    threshold = torch.quantile(all_confidences_tensor, target_ratio).item()

    return threshold


def collect_hard_examples(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Collect hard examples (low confidence samples) from validation set.

    Hard examples are defined as samples where the model's prediction
    confidence falls below the specified threshold. These examples are
    challenging for the shallow model and benefit from deeper processing.

    Args:
        model: Trained shallow model
        val_batches: Validation data batches
        threshold: Confidence threshold for identifying hard examples
        device: Device to run computation on

    Returns:
        Dictionary containing:
        - 'hidden_states': Layer outputs (used as input for deeper layers)
        - 'targets': Ground truth labels
    """
    model.eval()

    hard_hidden_states = []
    hard_targets = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)

            # Get hidden states using model method
            h = model.get_hidden_states(x)

            # Compute confidence
            _, confidence = model.compute_confidence(h)

            # Identify low-confidence samples
            mask = confidence < threshold

            if mask.any():
                # Flatten batch and sequence dimensions
                h_flat = h.view(-1, h.shape[-1])
                y_flat = y.view(-1)
                mask_flat = mask.view(-1)

                # Collect hard examples
                hard_hidden_states.append(h_flat[mask_flat])
                hard_targets.append(y_flat[mask_flat])

    if not hard_hidden_states:
        raise ValueError("No hard examples found. Try increasing threshold.")

    return {
        'hidden_states': torch.cat(hard_hidden_states),
        'targets': torch.cat(hard_targets),
    }


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
        # Hard examples are individual tokens (batch, dim).
        # Add seq_len=1 dimension for compatibility with forward_upper_layers.
        h_batch = hidden_states[batch_indices].unsqueeze(1)
        t_batch = targets[batch_indices]
        batches.append((h_batch, t_batch))

    return batches


def _forward_upper_on_hard_batch(
    model: nn.Module,
    h_batch: torch.Tensor,
    num_lower_layers: int,
    device: str
) -> torch.Tensor:
    """Forward through upper layers on a hard example batch.

    Hard examples are individual tokens extracted from sequences, stored as (batch, dim).
    Model's forward_upper_layers expects (batch, seq_len, dim), so we add seq_len=1.
    Output logits are (batch, 1, vocab), squeezed to (batch, vocab) for loss computation.

    Args:
        model: Model with forward_upper_layers method
        h_batch: Hidden states (batch_size, 1, dim) - seq_len dimension already added
        num_lower_layers: Number of lower layers to skip
        device: Device to run on

    Returns:
        Logits (batch_size, vocab_size)
    """
    h_batch = h_batch.to(device)
    # forward_upper_layers returns (batch, seq_len=1, vocab), squeeze to (batch, vocab)
    return model.forward_upper_layers(h_batch, num_lower_layers).squeeze(1)


def train_upper_layers(
    model: nn.Module,
    hard_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str,
    num_lower_layers: int = 2
) -> float:
    """
    Train upper layers on hard examples only.

    The lower layers are frozen (already trained in Phase 1).
    Only the newly added upper layers are trained on hard examples.

    Args:
        model: Extended model with upper layers
        hard_batches: Batches of (hidden_state, target) pairs
        optimizer: Optimizer for trainable parameters
        device: Device to run training on
        num_lower_layers: Number of frozen lower layers

    Returns:
        Average training loss for this epoch
    """
    model.train()
    total_loss = 0.0

    for h, y in hard_batches:
        y = y.to(device)
        optimizer.zero_grad()

        logits = _forward_upper_on_hard_batch(model, h, num_lower_layers, device)
        loss = F.cross_entropy(logits, y)

        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(hard_batches)


def _iterate_hard_batches(
    hard_examples: Dict[str, torch.Tensor],
    batch_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create sequential batches from hard examples (no shuffling).

    Unlike create_hard_example_loader, this preserves order for deterministic evaluation.

    Args:
        hard_examples: Dictionary with 'hidden_states' and 'targets'
        batch_size: Number of examples per batch

    Returns:
        List of (hidden_states, targets) batches with seq_len=1 dimension added
    """
    hidden_states = hard_examples['hidden_states']
    targets = hard_examples['targets']
    num_samples = len(targets)

    batches = []
    for i in range(0, num_samples, batch_size):
        h_batch = hidden_states[i:i + batch_size].unsqueeze(1)
        t_batch = targets[i:i + batch_size]
        batches.append((h_batch, t_batch))

    return batches


def evaluate_on_hard_examples(
    model: nn.Module,
    hard_examples: Dict[str, torch.Tensor],
    device: str,
    batch_size: int = 64,
    num_lower_layers: int = 2
) -> float:
    """
    Evaluate model performance on hard examples only.

    This measures how well the model handles the most challenging examples,
    which is crucial for understanding the benefit of deeper processing.

    Args:
        model: Model to evaluate (can be shallow or deep)
        hard_examples: Dictionary with 'hidden_states' and 'targets'
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_lower_layers: Number of lower layers (for deep model evaluation)

    Returns:
        Perplexity on hard examples
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    batches = _iterate_hard_batches(hard_examples, batch_size)

    with torch.no_grad():
        for h_batch, y_batch in batches:
            y_batch = y_batch.to(device)
            logits = _forward_upper_on_hard_batch(model, h_batch, num_lower_layers, device)
            loss = F.cross_entropy(logits, y_batch, reduction='sum')

            total_loss += loss.item()
            total_samples += len(y_batch)

    avg_loss = total_loss / total_samples
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl
