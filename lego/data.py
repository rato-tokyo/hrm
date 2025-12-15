"""
LEGO Framework - SequenceData

Data container for LEGO block training (sequence-based).
"""

import torch
from typing import Iterator, Tuple, Optional


class SequenceData:
    """
    Container for LEGO block training data (sequence-based).

    Holds sequences of (hidden_states, targets) for block training.
    Maintains sequence structure for proper Attention computation.

    Args:
        hidden_states: Hidden states tensor (num_sequences, seq_len, dim)
        targets: Target labels tensor (num_sequences, seq_len)

    Usage:
        # Create from tensors
        data = SequenceData(hidden_states, targets)

        # Iterate in batches (preserving sequences)
        for h, y in data.batches(batch_size=8):
            # h: (batch_size, seq_len, dim)
            # y: (batch_size, seq_len)
            ...

        # Split into train/val
        train_data, val_data = data.split(train_ratio=0.8)
    """

    def __init__(self, hidden_states: torch.Tensor, targets: torch.Tensor):
        if hidden_states.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: hidden_states={hidden_states.shape[0]}, targets={targets.shape[0]}"
            )
        if hidden_states.shape[1] != targets.shape[1]:
            raise ValueError(
                f"Seq len mismatch: hidden_states={hidden_states.shape[1]}, targets={targets.shape[1]}"
            )
        self._hidden_states = hidden_states  # (num_sequences, seq_len, dim)
        self._targets = targets  # (num_sequences, seq_len)

    @property
    def hidden_states(self) -> torch.Tensor:
        """Hidden states tensor (num_sequences, seq_len, dim)."""
        return self._hidden_states

    @property
    def targets(self) -> torch.Tensor:
        """Target labels tensor (num_sequences, seq_len)."""
        return self._targets

    @property
    def dim(self) -> int:
        """Hidden state dimension."""
        return self._hidden_states.shape[-1]

    @property
    def seq_len(self) -> int:
        """Sequence length."""
        return self._hidden_states.shape[1]

    @property
    def num_sequences(self) -> int:
        """Number of sequences."""
        return self._hidden_states.shape[0]

    @property
    def num_tokens(self) -> int:
        """Total number of tokens."""
        return self._hidden_states.shape[0] * self._hidden_states.shape[1]

    def __len__(self) -> int:
        """Number of sequences."""
        return self.num_sequences

    def to(self, device: str) -> "SequenceData":
        """Move data to specified device."""
        return SequenceData(
            self._hidden_states.to(device),
            self._targets.to(device)
        )

    def batches(
        self,
        batch_size: int,
        shuffle: bool
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over batched sequences for training.

        Args:
            batch_size: Number of sequences per batch
            shuffle: Whether to shuffle before batching

        Yields:
            Tuple of (hidden_states, targets)
            - hidden_states: (batch_size, seq_len, dim)
            - targets: (batch_size, seq_len)
        """
        num_sequences = len(self)
        if shuffle:
            indices = torch.randperm(num_sequences)
        else:
            indices = torch.arange(num_sequences)

        for i in range(0, num_sequences, batch_size):
            batch_indices = indices[i:i + batch_size]
            h_batch = self._hidden_states[batch_indices]  # (batch, seq_len, dim)
            t_batch = self._targets[batch_indices]  # (batch, seq_len)
            yield h_batch, t_batch

    def split(self, train_ratio: float) -> Tuple["SequenceData", "SequenceData"]:
        """
        Split into training and validation sets.

        Args:
            train_ratio: Ratio of data for training

        Returns:
            Tuple of (train_data, val_data)
        """
        num_sequences = len(self)
        num_train = int(num_sequences * train_ratio)

        indices = torch.randperm(num_sequences)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        train_data = SequenceData(
            self._hidden_states[train_indices],
            self._targets[train_indices]
        )
        val_data = SequenceData(
            self._hidden_states[val_indices],
            self._targets[val_indices]
        )

        return train_data, val_data

    @classmethod
    def empty(cls, seq_len: int, dim: int, device: Optional[str]) -> "SequenceData":
        """Create an empty SequenceData instance."""
        hidden_states = torch.empty(0, seq_len, dim)
        targets = torch.empty(0, seq_len, dtype=torch.long)
        if device:
            hidden_states = hidden_states.to(device)
            targets = targets.to(device)
        return cls(hidden_states, targets)
