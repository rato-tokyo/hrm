"""
CASCADEフレームワーク - Dataset操作

Hugging Face Datasetを直接使用するユーティリティ関数。
"""

from __future__ import annotations

import torch
from typing import Iterator, Tuple, Optional, Dict, Any, Union
from datasets import Dataset


def create_cascade_dataset(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> Dataset:
    """
    hidden_statesとlabelsからHF Datasetを作成。

    Args:
        hidden_states: (num_sequences, seq_len, dim)
        labels: (num_sequences, seq_len)

    Returns:
        Hugging Face Dataset
    """
    return Dataset.from_dict({
        'hidden_states': hidden_states.cpu().numpy().tolist(),
        'labels': labels.cpu().numpy().tolist(),
    })


def get_dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """
    Datasetの情報を取得。

    Args:
        dataset: Hugging Face Dataset

    Returns:
        num_sequences, seq_len, dim, num_tokensを含むDict
    """
    if len(dataset) == 0:
        return {
            'num_sequences': 0,
            'seq_len': 0,
            'dim': 0,
            'num_tokens': 0,
        }

    first = dataset[0]
    hidden = first['hidden_states']
    seq_len = len(hidden)
    dim = len(hidden[0]) if seq_len > 0 else 0

    return {
        'num_sequences': len(dataset),
        'seq_len': seq_len,
        'dim': dim,
        'num_tokens': len(dataset) * seq_len,
    }


def dataset_to_tensors(
    dataset: Dataset,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DatasetをTensorに変換。

    Args:
        dataset: Hugging Face Dataset
        device: 配置するデバイス

    Returns:
        (hidden_states, labels)のタプル
    """
    hidden_states = torch.tensor(dataset['hidden_states'])
    labels = torch.tensor(dataset['labels'], dtype=torch.long)

    if device:
        hidden_states = hidden_states.to(device)
        labels = labels.to(device)

    return hidden_states, labels


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Datasetからバッチを反復。

    Args:
        dataset: Hugging Face Dataset
        batch_size: バッチサイズ
        shuffle: シャッフルするか
        device: テンソルを配置するデバイス

    Yields:
        (hidden_states, labels)のタプル
    """
    num_sequences = len(dataset)
    if num_sequences == 0:
        return

    if shuffle:
        indices = torch.randperm(num_sequences).tolist()
    else:
        indices = list(range(num_sequences))

    for i in range(0, num_sequences, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = dataset.select(batch_indices)

        hidden_states = torch.tensor(batch['hidden_states'])
        labels = torch.tensor(batch['labels'], dtype=torch.long)

        if device:
            hidden_states = hidden_states.to(device)
            labels = labels.to(device)

        yield hidden_states, labels


def create_empty_dataset(seq_len: int, dim: int) -> Dataset:
    """
    空のDatasetを作成。

    Args:
        seq_len: シーケンス長
        dim: hidden state次元

    Returns:
        空のHugging Face Dataset
    """
    return Dataset.from_dict({
        'hidden_states': [],
        'labels': [],
    })
