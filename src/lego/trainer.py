"""
LEGO Framework - Trainer

Training and evaluation for LEGO models.
"""

from __future__ import annotations


class Trainer:
    """
    Trainer for LEGO models.

    Args:
        vocab_size: Vocabulary size
        device: Device to run on ('cpu' or 'cuda')
    """

    def __init__(self, vocab_size: int, device: str = 'cpu'):
        self.vocab_size = vocab_size
        self.device = device


