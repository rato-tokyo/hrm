"""
LEGO Framework - Configuration Classes

TrainerConfig: Training hyperparameters for train_block()
ExperimentConfig: Model architecture and experiment settings
"""

from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """
    Configuration for block training.

    Used by train_block() to configure training hyperparameters.
    All fields are required (no default values).

    Uses CALM-style exit classifier (cos_sim based, no training required).

    Attributes:
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        grad_clip: Gradient clipping value
        val_ratio: Ratio of data for validation
        hard_ratio: Ratio of tokens to collect as hard examples
        lr: Learning rate
        verbose: Print training progress
    """
    batch_size: int
    max_epochs: int
    patience: int
    grad_clip: float
    val_ratio: float
    hard_ratio: float
    lr: float
    verbose: bool


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
        block_layers: Layer counts per block (e.g., (2, 2) for 2 blocks with 2 layers each)
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
