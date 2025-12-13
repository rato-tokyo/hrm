"""
Test Helpers for LEGO Framework

Provides deterministic test utilities for reproducible testing.
"""

import sys
sys.path.insert(0, 'src')



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
