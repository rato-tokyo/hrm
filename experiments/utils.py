"""
EASE Experiments - Utilities

Data preparation and experiment configuration.
"""

import torch
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    # Data
    train_chars: int = 100000
    val_chars: int = 10000
    seq_len: int = 64
    batch_size: int = 32

    # Model
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 3

    # Training
    lr: float = 1e-3
    max_epochs: int = 50
    grad_clip: float = 1.0

    # Early stopping
    patience: int = 0  # 0 = stop immediately when val worsens

    # Device
    device: str = 'cpu'

    # Random seed
    seed: int = 42


def prepare_wikitext_data(
    train_chars: int = 100000,
    val_chars: int = 10000,
    seq_len: int = 64,
    batch_size: int = 32,
    data_dir: str = 'data'
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
           List[Tuple[torch.Tensor, torch.Tensor]],
           int,
           Dict[str, int]]:
    """
    Prepare WikiText-2 character-level data.

    Returns:
        train_batches: List of (x, y) tensor tuples
        val_batches: List of (x, y) tensor tuples
        vocab_size: Size of vocabulary
        char_to_idx: Character to index mapping
    """
    try:
        with open(f'{data_dir}/wikitext2_train.txt', 'r') as f:
            train_text = f.read()[:train_chars]
        with open(f'{data_dir}/wikitext2_valid.txt', 'r') as f:
            val_text = f.read()[:val_chars]
    except FileNotFoundError:
        # Fallback for testing
        train_text = "The quick brown fox jumps over the lazy dog. " * (train_chars // 45 + 1)
        train_text = train_text[:train_chars]
        val_text = "A quick brown dog runs in the park. " * (val_chars // 35 + 1)
        val_text = val_text[:val_chars]

    # Build vocabulary from both train and val
    chars = sorted(set(train_text + val_text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    def text_to_batches(text: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        indices = [char_to_idx.get(c, 0) for c in text]
        batches = []

        for i in range(0, len(indices) - seq_len - 1, seq_len * batch_size):
            batch_x, batch_y = [], []
            for j in range(batch_size):
                start = i + j * seq_len
                if start + seq_len + 1 <= len(indices):
                    batch_x.append(indices[start:start + seq_len])
                    batch_y.append(indices[start + 1:start + seq_len + 1])
            if len(batch_x) == batch_size:
                batches.append((
                    torch.tensor(batch_x, dtype=torch.long),
                    torch.tensor(batch_y, dtype=torch.long)
                ))
        return batches

    train_batches = text_to_batches(train_text)
    val_batches = text_to_batches(val_text)

    return train_batches, val_batches, vocab_size, char_to_idx
