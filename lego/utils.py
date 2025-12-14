"""
LEGO Framework - Utility Functions

Basic utilities for LEGO training.

Note: Hard example collection and splitting are handled by:
- train_block() - Train block and return hard examples
- SequenceData.split() - Split into train/val
- SequenceData.batches() - Create batched data
"""

import random
import numpy as np
import torch
from typing import List, Tuple


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
    num_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic synthetic data for testing."""
    batches = []
    for i in range(num_batches):
        torch.manual_seed(42 + i)  # Deterministic per batch
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        batches.append((x, y))
    return batches
