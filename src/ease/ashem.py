"""
LEGO Framework - ASHEM Training Strategy

ASHEM: Adaptive Supervision via Hard Example Mining

A two-phase training strategy that focuses computational resources on hard examples:
1. Phase 1: Train shallow model on all data
2. Phase 2: Extend model with additional layers, train only on hard examples
3. Inference: Two-stage routing using LEGO's Early Exit mechanism

Key Benefits:
- 78% improvement on hard examples (PPL: 2600 → 571)
- 36% compute cost reduction using adaptive routing
- Fully integrated with LEGO framework's 3 core options

References:
- ASHEM: Adaptive Supervision via Hard Example Mining (本研究)
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- Early Exit: BranchyNet (2016), Teerapittayanon et al. (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ASHEMConfig:
    """
    Configuration for ASHEM (Adaptive Supervision via Hard Example Mining).

    ASHEM is a two-phase training strategy:
    1. Phase 1: Train shallow model on all data
    2. Phase 2: Extend model with additional layers, train only on hard examples

    Args:
        phase1_layers: Number of layers for shallow model in Phase 1
        phase1_lr: Learning rate for Phase 1 training
        phase1_patience: Early stopping patience for Phase 1 (default: 1)
        hard_example_ratio: Target ratio of hard examples to collect (0.0-1.0).
                           e.g., 0.5 means target 50% of examples as hard
        phase2_layers: Total number of layers after extension (must be > phase1_layers)
        phase2_lr: Learning rate for Phase 2 (typically lower for fine-tuning)
        phase2_patience: Early stopping patience for Phase 2 (default: 3)
    """
    # Phase 1: Shallow model
    phase1_layers: int = 2
    phase1_lr: float = 1e-3
    phase1_patience: int = 1

    # Hard example collection
    hard_example_ratio: float = 0.5  # Target 50% as hard examples

    # Phase 2: Deep model
    phase2_layers: int = 4  # Total layers
    phase2_lr: float = 1e-4  # Lower LR for fine-tuning
    phase2_patience: int = 3  # Higher patience for new layers


# ==============================================================================
# ASHEM Utility Functions
# ==============================================================================

def compute_confidence(model: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Compute prediction confidence from hidden state.

    Confidence is defined as the maximum probability in the softmax distribution.
    Higher confidence indicates the model is more certain about its prediction.

    Args:
        model: Model with output_head attribute
        hidden_state: Hidden state tensor of shape (batch_size, seq_len, dim)

    Returns:
        Confidence values of shape (batch_size, seq_len), range [0, 1]
    """
    logits = model.output_head(hidden_state)
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


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
        for x, y in val_batches:
            x = x.to(device)

            # Forward pass through all layers
            h = model.embedding(x)
            for layer in model.layers:
                h = layer(h)

            # Compute confidence
            confidence = compute_confidence(model, h)
            all_confidences.append(confidence.view(-1))

    # Concatenate all confidences and compute threshold
    all_confidences_tensor = torch.cat(all_confidences)
    threshold = torch.quantile(all_confidences_tensor, target_ratio).item()

    return threshold


def collect_hard_examples(
    model: nn.Module,
    data_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: str
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Collect hard examples (low confidence sequences) from dataset.

    Hard examples are sequences containing at least one token where the model's
    prediction confidence falls below the specified threshold. Returns full
    sequences (not individual tokens) to maintain compatibility with standard
    LEGO training workflow.

    Args:
        model: Trained shallow model
        data_batches: Data batches (train or val)
        threshold: Confidence threshold for identifying hard examples
        device: Device to run computation on

    Returns:
        List of (input, target) batches containing only hard sequences
    """
    model.eval()

    hard_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        for x, y in data_batches:
            x, y = x.to(device), y.to(device)

            # Forward pass through all layers
            h = model.embedding(x)
            for layer in model.layers:
                h = layer(h)

            # Compute confidence per token
            confidence = compute_confidence(model, h)

            # Identify sequences with at least one hard token
            # Shape: (batch_size,) - True if any token in sequence is hard
            hard_seq_mask = (confidence < threshold).any(dim=1)

            if hard_seq_mask.any():
                # Keep only hard sequences
                hard_x = x[hard_seq_mask]
                hard_y = y[hard_seq_mask]
                hard_batches.append((hard_x.cpu(), hard_y.cpu()))

    if not hard_batches:
        raise ValueError("No hard examples found. Try increasing threshold.")

    return hard_batches


def freeze_lower_layers(model: nn.Module, num_lower_layers: int) -> None:
    """
    Freeze embedding and lower layers for Phase 2 training.

    In LEGO's ASHEM strategy, lower layers (Block 1) are frozen after Phase 1,
    and only upper layers (Block 2) are trained on hard examples.

    Args:
        model: Model to freeze lower layers
        num_lower_layers: Number of layers to freeze (including embedding)
    """
    # Freeze embedding
    for param in model.embedding.parameters():
        param.requires_grad = False

    # Freeze lower layers
    for i in range(num_lower_layers):
        for param in model.layers[i].parameters():
            param.requires_grad = False


def get_trainable_params_info(model: nn.Module) -> Dict[str, int]:
    """
    Get information about trainable and total parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with 'trainable', 'total', and 'ratio' keys
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'total': total,
        'ratio': trainable / total if total > 0 else 0.0
    }
