"""
LEGO Framework - Data Utilities

WikiText-2 dataloader and LearningData class for LEGO training.
"""

import torch
from typing import Dict, Iterator, List, Tuple, Optional


class TrainingData:
    """
    Container for LEGO block training data.

    Holds (hidden_states, target) pairs collected from hard examples.
    Provides iteration, batching, and train/val split functionality.

    Args:
        hidden_states: Hidden states tensor (num_tokens, dim)
        targets: Target labels tensor (num_tokens,)

    Usage:
        # Create from tensors
        data = TrainingData(hidden_states, targets)

        # Iterate in batches
        for h, y in data.batches(batch_size=64):
            ...

        # Split into train/val
        train_data, val_data = data.split(train_ratio=0.8)

        # Get statistics
        print(f"Tokens: {len(data)}, Dim: {data.dim}")
    """

    def __init__(self, hidden_states: torch.Tensor, targets: torch.Tensor):
        if len(hidden_states) != len(targets):
            raise ValueError(
                f"Length mismatch: hidden_states={len(hidden_states)}, targets={len(targets)}"
            )
        self._hidden_states = hidden_states
        self._targets = targets

    @property
    def hidden_states(self) -> torch.Tensor:
        """Hidden states tensor (num_tokens, dim)."""
        return self._hidden_states

    @property
    def targets(self) -> torch.Tensor:
        """Target labels tensor (num_tokens,)."""
        return self._targets

    @property
    def dim(self) -> int:
        """Hidden state dimension."""
        return self._hidden_states.shape[-1]

    def __len__(self) -> int:
        """Number of tokens."""
        return len(self._targets)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over individual (hidden_state, target) pairs."""
        for i in range(len(self)):
            yield self._hidden_states[i], self._targets[i]

    def to(self, device: str) -> "TrainingData":
        """Move data to specified device."""
        return TrainingData(
            self._hidden_states.to(device),
            self._targets.to(device)
        )

    def batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over batched data for training.

        Args:
            batch_size: Number of tokens per batch
            shuffle: Whether to shuffle before batching

        Yields:
            Tuple of (hidden_states, targets)
            - hidden_states: (batch_size, 1, dim) for LEGOBlock.forward()
            - targets: (batch_size,)
        """
        num_samples = len(self)
        if shuffle:
            indices = torch.randperm(num_samples)
        else:
            indices = torch.arange(num_samples)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            # Add seq_len=1 dimension for LEGOBlock.forward()
            h_batch = self._hidden_states[batch_indices].unsqueeze(1)
            t_batch = self._targets[batch_indices]
            yield h_batch, t_batch

    def split(self, train_ratio: float = 0.8) -> Tuple["TrainingData", "TrainingData"]:
        """
        Split into training and validation sets.

        Args:
            train_ratio: Ratio of data for training (default: 0.8)

        Returns:
            Tuple of (train_data, val_data)
        """
        num_samples = len(self)
        num_train = int(num_samples * train_ratio)

        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        train_data = TrainingData(
            self._hidden_states[train_indices],
            self._targets[train_indices]
        )
        val_data = TrainingData(
            self._hidden_states[val_indices],
            self._targets[val_indices]
        )

        return train_data, val_data

    @classmethod
    def empty(cls, dim: int, device: Optional[str] = None) -> "TrainingData":
        """Create an empty TrainingData instance."""
        hidden_states = torch.empty(0, dim)
        targets = torch.empty(0, dtype=torch.long)
        if device:
            hidden_states = hidden_states.to(device)
            targets = targets.to(device)
        return cls(hidden_states, targets)


def create_wikitext_dataloaders(
    num_samples: int,
    batch_size: int,
    seq_len: int = 32,
    seed: int = 42
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
           List[Tuple[torch.Tensor, torch.Tensor]],
           int]:
    """
    Create WikiText-2 dataloaders for language modeling.

    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        seq_len: Sequence length (default: 32)
        seed: Random seed (default: 42)

    Returns:
        Tuple of (train_batches, val_batches, vocab_size)
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Please install datasets: pip install datasets") from exc

    torch.manual_seed(seed)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def simple_tokenize(text: str) -> List[str]:
        return text.lower().split()

    # Build vocabulary
    vocab: Dict[str, int] = {'<unk>': 0, '<pad>': 1}
    for split in ['train', 'validation']:
        for item in dataset[split]:
            for token in simple_tokenize(item['text']):
                if token not in vocab:
                    vocab[token] = len(vocab)

    vocab_size = len(vocab)

    def tokenize_split(split_name: str) -> torch.Tensor:
        all_tokens: List[int] = []
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
                batches.append((torch.stack(batch_x), torch.stack(batch_y)))
        return batches

    return batchify(train_data), batchify(val_data), vocab_size
