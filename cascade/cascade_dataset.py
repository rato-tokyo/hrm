"""
CASCADEフレームワーク - Dataset操作

Hugging Face Datasetを直接使用するユーティリティ関数。
SequenceDataの代替として、HF Datasetをそのまま活用。
"""

from __future__ import annotations

import torch
from typing import Iterator, Tuple, Optional, Dict, Any, List, TYPE_CHECKING, Union
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

    レイヤー数に応じて適切なcos_sim計算を行う:
    - 30層以上: Layer 28-29の平均cos_sim（最終レイヤーは出力変換で不適切）
    - 4層以上: 最後から2-3番目のレイヤーの平均cos_sim
    - 2-3層: 最後から1-2番目のcos_sim
    - 1層: 入力と出力のcos_sim

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
    llm.eval()

    info = get_dataset_info(dataset)
    if info['num_sequences'] == 0:
        return create_empty_dataset(info['seq_len'], info['dim']), 0.0

    all_avg_cos_sim: List[torch.Tensor] = []
    all_hidden_out: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for h, y in iterate_batches(dataset, batch_size, shuffle=False, device=device):
            # dtype変換はLLM.forwardで自動実行
            h_out, hidden_history = llm.forward(h, input_type="hidden_states")

            # レイヤー数に応じたcos_sim計算
            num_states = len(hidden_history)  # 入力 + 各レイヤー出力

            if num_states >= 5:
                # 4層以上: 最後から2-3番目のレイヤーの平均
                h_prev2 = hidden_history[-4]
                h_prev1 = hidden_history[-3]
                h_last = hidden_history[-2]
                cos_sim_1 = compute_cos_sim(h_prev2, h_prev1)
                cos_sim_2 = compute_cos_sim(h_prev1, h_last)
                avg_cos_sim = (cos_sim_1 + cos_sim_2) / 2.0
            elif num_states >= 3:
                # 2-3層: 最後から1-2番目のcos_sim
                h_prev = hidden_history[-3]
                h_last = hidden_history[-2]
                avg_cos_sim = compute_cos_sim(h_prev, h_last)
            else:
                # 1層: 入力と出力のcos_sim
                h_in = hidden_history[0]
                h_out_layer = hidden_history[-2] if num_states > 1 else hidden_history[-1]
                avg_cos_sim = compute_cos_sim(h_in, h_out_layer)

            all_avg_cos_sim.append(avg_cos_sim.cpu())
            all_hidden_out.append(h_out.cpu())
            all_targets.append(y.cpu())

    avg_cos_sim_all = torch.cat(all_avg_cos_sim)
    hidden_out_all = torch.cat(all_hidden_out)
    targets_all = torch.cat(all_targets)

    # 閾値を計算（float32に変換してquantile計算）
    all_cos_flat = avg_cos_sim_all.view(-1).float()
    if hard_ratio >= 1.0:
        threshold = float('inf')
    elif hard_ratio <= 0.0:
        threshold = float('-inf')
    else:
        threshold = float(torch.quantile(all_cos_flat, hard_ratio).item())

    # トークン単位のhardマスク
    hard_token_mask = avg_cos_sim_all < threshold
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
    llm.eval()

    info = get_dataset_info(dataset)
    if info['num_sequences'] == 0:
        return create_empty_dataset(info['seq_len'], info['dim'])

    all_hidden: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for h, y in iterate_batches(dataset, batch_size, shuffle=False, device=device):
            # dtype変換はLLM.forwardで自動実行
            h_out, _ = llm.forward(h, input_type="hidden_states")
            all_hidden.append(h_out.cpu())
            all_targets.append(y.cpu())

    return create_cascade_dataset(torch.cat(all_hidden), torch.cat(all_targets))
