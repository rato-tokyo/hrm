"""
LASH Experiments - Utilities

Data preparation and experiment configuration for LASH framework experiments.
"""

import torch
import random
import numpy as np
from typing import List, Tuple, Dict


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


def load_wikitext_tokenized(seq_len: int = 32, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load WikiText-2 data and tokenize (word-level).

    Returns:
        train_data: Tokenized training data
        val_data: Tokenized validation data
        vocab_size: Vocabulary size
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    torch.manual_seed(seed)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # Simple word tokenizer
    def simple_tokenize(text):
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
    def tokenize_split(split_name):
        all_tokens = []
        for item in dataset[split_name]:
            tokens = simple_tokenize(item['text'])
            token_ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
            all_tokens.extend(token_ids)
        return torch.tensor(all_tokens, dtype=torch.long)

    train_data = tokenize_split('train')
    val_data = tokenize_split('validation')

    return train_data, val_data, vocab_size


def batchify(data: torch.Tensor, batch_size: int, seq_len: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create batches from tokenized data.

    Args:
        data: Tokenized data tensor
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        List of (input, target) batches
    """
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
    train_data, val_data, vocab_size = load_wikitext_tokenized(seq_len, seed)

    # Limit samples
    max_tokens_train = num_samples * (seq_len + 1)
    max_tokens_val = int(num_samples * 0.2) * (seq_len + 1)

    train_data = train_data[:max_tokens_train]
    val_data = val_data[:max_tokens_val]

    train_loader = batchify(train_data, batch_size, seq_len)
    val_loader = batchify(val_data, batch_size, seq_len)

    return train_loader, val_loader, vocab_size
