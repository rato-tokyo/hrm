"""
CASCADEフレームワーク - Dataset操作

Hugging Face Datasetを直接使用するユーティリティ関数。
SequenceDataの代替として、HF Datasetをそのまま活用。
"""

from __future__ import annotations

import torch
from typing import Iterator, Tuple, Optional, Dict, Any, List, TYPE_CHECKING
from datasets import Dataset

if TYPE_CHECKING:
    from .llm import LLM


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
    device: Optional[str] = None,
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


def reconstruct_sequences(
    hidden_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    seq_len: int,
) -> Dataset:
    """
    フラットなトークンをシーケンスに再構成してDatasetを作成。

    Args:
        hidden_flat: (num_tokens, dim)
        labels_flat: (num_tokens,)
        seq_len: シーケンス長

    Returns:
        再構成されたHugging Face Dataset
    """
    num_tokens = hidden_flat.shape[0]
    num_complete = num_tokens // seq_len

    if num_complete == 0:
        return create_empty_dataset(seq_len, hidden_flat.shape[-1] if num_tokens > 0 else 0)

    usable = num_complete * seq_len
    hidden_seq = hidden_flat[:usable].view(num_complete, seq_len, -1)
    labels_seq = labels_flat[:usable].view(num_complete, seq_len)

    return create_cascade_dataset(hidden_seq, labels_seq)


def collect_hard_tokens_from_dataset(
    llm: "LLM",
    dataset: Dataset,
    hard_ratio: float,
    batch_size: int,
) -> Tuple[Dataset, float]:
    """
    Datasetからhard tokensを収集。

    Args:
        llm: 評価に使用するLLM
        dataset: 入力Dataset
        hard_ratio: hard tokenの割合 (0.0-1.0)
        batch_size: バッチサイズ

    Returns:
        (hard_dataset, threshold)のタプル
    """
    from .exit_fn import compute_cos_sim

    device = next(llm.parameters()).device
    # モデルのdtypeを取得（float16対応）
    model_dtype = next(llm.parameters()).dtype
    llm.eval()

    info = get_dataset_info(dataset)
    if info['num_sequences'] == 0:
        return create_empty_dataset(info['seq_len'], info['dim']), 0.0

    all_cos_sim: List[torch.Tensor] = []
    all_hidden_out: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for h, y in iterate_batches(dataset, batch_size, shuffle=False, device=str(device)):
            # モデルのdtypeに合わせる（float16対応）
            h = h.to(dtype=model_dtype)
            h_out, hidden_history = llm.forward(h, input_type="hidden_states")
            h_in = hidden_history[-2]
            cos_sim = compute_cos_sim(h_in, h_out)

            all_cos_sim.append(cos_sim.cpu())
            all_hidden_out.append(h_out.cpu())
            all_targets.append(y.cpu())

    cos_sim_all = torch.cat(all_cos_sim)
    hidden_out_all = torch.cat(all_hidden_out)
    targets_all = torch.cat(all_targets)

    # 閾値を計算
    all_cos_flat = cos_sim_all.view(-1)
    if hard_ratio >= 1.0:
        threshold = float('inf')
    elif hard_ratio <= 0.0:
        threshold = float('-inf')
    else:
        threshold = float(torch.quantile(all_cos_flat, hard_ratio).item())

    # トークン単位のhardマスク
    hard_token_mask = cos_sim_all < threshold
    hard_hidden = hidden_out_all[hard_token_mask]
    hard_targets = targets_all[hard_token_mask]

    # シーケンスに再構成
    hard_dataset = reconstruct_sequences(hard_hidden, hard_targets, info['seq_len'])

    return hard_dataset, threshold


def transform_dataset(
    llm: "LLM",
    dataset: Dataset,
    batch_size: int,
) -> Dataset:
    """
    DatasetをLLMで変換。

    Args:
        llm: 変換に使用するLLM
        dataset: 入力Dataset
        batch_size: バッチサイズ

    Returns:
        変換されたDataset
    """
    device = next(llm.parameters()).device
    # モデルのdtypeを取得（float16対応）
    model_dtype = next(llm.parameters()).dtype
    llm.eval()

    info = get_dataset_info(dataset)
    if info['num_sequences'] == 0:
        return create_empty_dataset(info['seq_len'], info['dim'])

    all_hidden: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for h, y in iterate_batches(dataset, batch_size, shuffle=False, device=str(device)):
            # モデルのdtypeに合わせる（float16対応）
            h = h.to(dtype=model_dtype)
            h_out, _ = llm.forward(h, input_type="hidden_states")
            all_hidden.append(h_out.cpu())
            all_targets.append(y.cpu())

    return create_cascade_dataset(torch.cat(all_hidden), torch.cat(all_targets))
