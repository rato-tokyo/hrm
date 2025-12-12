"""
LEGO Experiments - Utilities

Data preparation utilities for LEGO framework experiments.
"""

import torch
import random
import numpy as np
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


def create_wikitext_dataloaders(
    num_samples: int,
    batch_size: int,
    seq_len: int = 32,
    seed: int = 42
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
           List[Tuple[torch.Tensor, torch.Tensor]],
           int]:
    """
    Create WikiText-2 dataloaders for experiments.

    Args:
        num_samples: Number of training samples to use
        batch_size: Batch size
        seq_len: Sequence length
        seed: Random seed

    Returns:
        train_loader: Training batches
        val_loader: Validation batches
        vocab_size: Vocabulary size
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    torch.manual_seed(seed)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # Simple word tokenizer
    def simple_tokenize(text: str) -> list[str]:
        return text.lower().split()

    # Build vocabulary
    vocab = {'<unk>': 0, '<pad>': 1}
    for split in ['train', 'validation']:
        for item in dataset[split]:
            tokens = simple_tokenize(item['text'])
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

    vocab_size = len(vocab)

    # Tokenize data
    def tokenize_split(split_name: str) -> torch.Tensor:
        all_tokens: list[int] = []
        for item in dataset[split_name]:
            tokens = simple_tokenize(item['text'])
            token_ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
            all_tokens.extend(token_ids)
        return torch.tensor(all_tokens, dtype=torch.long)

    train_data = tokenize_split('train')
    val_data = tokenize_split('validation')

    # Limit samples
    max_tokens_train = num_samples * (seq_len + 1)
    max_tokens_val = int(num_samples * 0.2) * (seq_len + 1)

    train_data = train_data[:max_tokens_train]
    val_data = val_data[:max_tokens_val]

    # Create batches
    def batchify(data: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batches = []
        num_tokens = len(data)

        for i in range(0, num_tokens - seq_len - 1, batch_size * seq_len):
            batch_x, batch_y = [], []
            for j in range(batch_size):
                start_idx = i + j * seq_len
                if start_idx + seq_len + 1 <= num_tokens:
                    batch_x.append(data[start_idx:start_idx + seq_len])
                    batch_y.append(data[start_idx + 1:start_idx + seq_len + 1])

            if len(batch_x) == batch_size:
                batches.append((
                    torch.stack(batch_x),
                    torch.stack(batch_y)
                ))

        return batches

    train_loader = batchify(train_data)
    val_loader = batchify(val_data)

    return train_loader, val_loader, vocab_size
