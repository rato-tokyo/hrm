"""
Test Helpers for LEGO Framework

Provides deterministic test utilities for reproducible testing.
"""

import torch
from typing import List, Tuple


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_data(
    num_batches: int = 4,
    batch_size: int = 8,
    seq_len: int = 16,
    vocab_size: int = 100
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic synthetic data."""
    batches = []
    for i in range(num_batches):
        torch.manual_seed(42 + i)  # Deterministic per batch
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        batches.append((x, y))
    return batches


def assert_close(actual: float, expected: float, name: str, rtol: float = 1e-4) -> None:
    """Assert two values are close with detailed error message."""
    if abs(actual - expected) > abs(expected) * rtol + 1e-8:
        raise AssertionError(
            f"{name}: MISMATCH\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  Diff:     {abs(actual - expected)}"
        )
    print(f"  {name}: {actual:.6f} (expected {expected:.6f})")
