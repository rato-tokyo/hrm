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
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 3
    grad_clip: float = 1.0
    val_ratio: float = 0.2
    hard_ratio: float = 0.5
    lr: float = 1e-3
    verbose: bool = True


@dataclass
class ExperimentConfig:
    """
    Configuration for LEGO experiment.

    Model architecture and data settings.

    Attributes:
        dim: Model dimension
        num_heads: Number of attention heads
        seq_len: Sequence length for language modeling
        num_samples: Number of training samples
        block_layers: List of layer counts per block (e.g., [2, 2] for 2 blocks)
    """
    # Model architecture
    dim: int = 64
    num_heads: int = 4
    seq_len: int = 32

    # Dataset
    num_samples: int = 10000

    # Block configuration
    block_layers: tuple[int, ...] = (2, 2)  # Layers per block
