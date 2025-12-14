"""
LEGO Framework - Utility Functions
"""

import random
import numpy as np
import torch


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
