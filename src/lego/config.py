"""
LEGO Framework - Experiment Configuration

Default configuration for LEGO two-phase training experiments.
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """
    Configuration for LEGO experiment.

    Attributes:
        seq_len: Sequence length for language modeling
        dim: Model dimension
        num_heads: Number of attention heads
        phase1_samples: Number of training samples for Phase 1
        phase1_batch: Batch size for Phase 1
        phase2_batch: Batch size for Phase 2
        phase1_epochs: Maximum epochs for Phase 1
        phase2_epochs: Maximum epochs for Phase 2
        phase1_layers: Number of layers in Phase 1 (shallow model)
        phase1_lr: Learning rate for Phase 1
        phase1_patience: Early stopping patience for Phase 1
        hard_example_ratio: Target ratio of hard examples (0.0-1.0)
        phase2_layers: Total number of layers after extension
        phase2_lr: Learning rate for Phase 2
        phase2_patience: Early stopping patience for Phase 2
    """
    # Model architecture
    seq_len: int = 32
    dim: int = 64
    num_heads: int = 4

    # Dataset parameters
    phase1_samples: int = 10000
    phase1_batch: int = 64
    phase2_batch: int = 64
    phase1_epochs: int = 50
    phase2_epochs: int = 50

    # Phase 1: Shallow model
    phase1_layers: int = 2
    phase1_lr: float = 1e-3
    phase1_patience: int = 1

    # Hard example collection
    hard_example_ratio: float = 0.5

    # Phase 2: Deep model
    phase2_layers: int = 4
    phase2_lr: float = 1e-4
    phase2_patience: int = 3
