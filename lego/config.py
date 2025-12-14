"""
LEGO Framework - Configuration Classes

TrainerConfig: Training hyperparameters for train_block()
ExperimentConfig: Model architecture and experiment settings
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainerConfig:
    """
    Configuration for block training.

    Used by train_block() to configure training hyperparameters.
    All fields are required (no default values).

    Attributes:
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        grad_clip: Gradient clipping value
        val_ratio: Ratio of data for validation
        hard_ratio: Ratio of tokens to collect as hard examples
        lr: Learning rate
        verbose: Print training progress
        exit_classifier_mode: How to train exit_classifier
            - "joint": Train with LM loss in same loop (BCE loss added)
            - "post": Train separately after LM training completes
        exit_label_mode: What label to use for exit_classifier training
            - "correct": 1 if prediction is correct, 0 otherwise (binary)
            - "distill": softmax confidence as continuous target (regression)
            - "loss": negative cross-entropy loss as target (regression)
    """
    batch_size: int
    max_epochs: int
    patience: int
    grad_clip: float
    val_ratio: float
    hard_ratio: float
    lr: float
    verbose: bool
    exit_classifier_mode: Literal["joint", "post"]
    exit_label_mode: Literal["correct", "distill", "loss"]


@dataclass
class ExperimentConfig:
    """
    Configuration for LEGO experiment.

    Model architecture and data settings.
    All fields are required (no default values).

    Attributes:
        dim: Model dimension
        num_heads: Number of attention heads
        ffn_dim: FFN hidden dimension
        max_seq_len: Maximum sequence length
        causal: Whether to use causal masking
        eps: Epsilon for RMSNorm
        seq_len: Sequence length for language modeling
        num_samples: Number of training samples
        block_layers: List of layer counts per block (e.g., [2, 2] for 2 blocks)
    """
    dim: int
    num_heads: int
    ffn_dim: int
    max_seq_len: int
    causal: bool
    eps: float
    seq_len: int
    num_samples: int
    block_layers: tuple[int, ...]
