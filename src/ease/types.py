"""
LEGO Framework - Type Definitions

Shared type definitions for the LEGO framework.
"""

import torch
from typing import Tuple, List, Optional
from typing_extensions import TypedDict


# ==============================================================================
# Data Batch Types
# ==============================================================================

# Standard data batch: (input_ids, target_ids)
DataBatch = Tuple[torch.Tensor, torch.Tensor]

# Hard example batch: (hidden_states, targets) - used in Phase 2+ training
HardBatch = Tuple[torch.Tensor, torch.Tensor]


# ==============================================================================
# Hard Examples
# ==============================================================================

class HardExamples(TypedDict):
    """
    Collection of hard examples for deeper phase training.

    Hard examples are tokens where the shallow model has low confidence.
    These benefit most from deeper processing.

    Attributes:
        inputs: Original input token IDs (num_hard_tokens,)
        hidden_states: Layer outputs from shallow model (num_hard_tokens, dim)
        targets: Ground truth labels (num_hard_tokens,)
        confidences: Confidence scores at collection time (num_hard_tokens,)
    """
    inputs: torch.Tensor
    hidden_states: torch.Tensor
    targets: torch.Tensor
    confidences: torch.Tensor


# ==============================================================================
# Evaluation Statistics
# ==============================================================================

class EvalStats(TypedDict):
    """
    Evaluation statistics returned by Trainer.evaluate().

    Attributes:
        ppl: Perplexity (lower is better)
        acc: Accuracy (0.0-1.0)
        shallow_ratio: Fraction of tokens using early exit (0.0-1.0)
        compute_cost: Relative compute cost (1.0 = full model)
    """
    ppl: float
    acc: float
    shallow_ratio: float
    compute_cost: float


# ==============================================================================
# Training History Types
# ==============================================================================

class TrainingHistory(TypedDict):
    """
    Training history returned by Trainer.train_with_early_stopping().

    Attributes:
        train_losses: Training loss per epoch
        val_losses: Validation PPL per epoch
        val_accs: Validation accuracy per epoch
        best_epoch: Epoch with best validation loss (0-indexed)
        total_epochs: Total epochs trained
        stopped_early: Whether early stopping was triggered
    """
    train_losses: List[float]
    val_losses: List[float]
    val_accs: List[float]
    best_epoch: int
    total_epochs: int
    stopped_early: bool


class PhaseHistory(TypedDict):
    """
    Training history for a single LEGO phase.

    Attributes:
        train_losses: Training loss per epoch
        val_ppls: Validation PPL per epoch
        best_epoch: Epoch with best validation performance (0-indexed)
        total_epochs: Total epochs trained
    """
    train_losses: List[float]
    val_ppls: List[float]
    best_epoch: int
    total_epochs: int


class LEGOResult(TypedDict):
    """
    Result from LEGOTrainer.train().

    Attributes:
        thresholds: Confidence thresholds for each phase transition
        phase_histories: Training history for each phase
        hard_examples: Final hard examples (from last phase, if applicable)
    """
    thresholds: List[float]
    phase_histories: List[PhaseHistory]
    hard_examples: Optional[HardExamples]
