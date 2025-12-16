"""
CASCADEフレームワーク - SequenceData

Hugging Face Datasetをベースにした LLM訓練用データコンテナ。
hidden states とターゲットのシーケンスを保持。
"""

import torch
from typing import Iterator, Tuple, Optional, List
from datasets import Dataset


class SequenceData:
    """
    Hugging Face DatasetベースのLLM訓練用データコンテナ。

    hidden_states と targets のシーケンスを保持し、
    Hugging Face Dataset の機能（シャッフル、分割等）を活用。

    Args:
        hidden_states: hidden statesテンソル (num_sequences, seq_len, dim)
        targets: ターゲットラベルテンソル (num_sequences, seq_len)

    使用例:
        # テンソルから作成
        data = SequenceData(hidden_states, targets)

        # バッチで反復（シーケンスを維持）
        for h, y in data.batches(batch_size=8, shuffle=True):
            # h: (batch_size, seq_len, dim)
            # y: (batch_size, seq_len)
            ...

        # train/valに分割
        train_data, val_data = data.split(train_ratio=0.8)

        # Hugging Face Datasetとしてアクセス
        hf_dataset = data.to_hf_dataset()
    """

    def __init__(self, hidden_states: torch.Tensor, targets: torch.Tensor):
        if hidden_states.shape[0] != targets.shape[0]:
            raise ValueError(
                f"バッチサイズ不一致: hidden_states={hidden_states.shape[0]}, targets={targets.shape[0]}"
            )
        if len(hidden_states.shape) >= 2 and len(targets.shape) >= 2:
            if hidden_states.shape[1] != targets.shape[1]:
                raise ValueError(
                    f"シーケンス長不一致: hidden_states={hidden_states.shape[1]}, targets={targets.shape[1]}"
                )

        self._hidden_states = hidden_states  # (num_sequences, seq_len, dim)
        self._targets = targets  # (num_sequences, seq_len)
        self._dataset: Optional[Dataset] = None

    @property
    def hidden_states(self) -> torch.Tensor:
        """hidden statesテンソル (num_sequences, seq_len, dim)。"""
        return self._hidden_states

    @property
    def targets(self) -> torch.Tensor:
        """ターゲットラベルテンソル (num_sequences, seq_len)。"""
        return self._targets

    @property
    def dim(self) -> int:
        """hidden state次元。"""
        return self._hidden_states.shape[-1]

    @property
    def seq_len(self) -> int:
        """シーケンス長。"""
        return self._hidden_states.shape[1] if len(self._hidden_states.shape) > 1 else 0

    @property
    def num_sequences(self) -> int:
        """シーケンス数。"""
        return self._hidden_states.shape[0]

    @property
    def num_tokens(self) -> int:
        """総トークン数。"""
        if len(self._hidden_states.shape) < 2:
            return 0
        return self._hidden_states.shape[0] * self._hidden_states.shape[1]

    def __len__(self) -> int:
        """シーケンス数。"""
        return self.num_sequences

    def to(self, device: str) -> "SequenceData":
        """指定デバイスにデータを移動。"""
        return SequenceData(
            self._hidden_states.to(device),
            self._targets.to(device)
        )

    def to_hf_dataset(self) -> Dataset:
        """Hugging Face Datasetに変換。"""
        if self._dataset is None:
            # テンソルをリストに変換（Datasetの要件）
            self._dataset = Dataset.from_dict({
                'hidden_states': self._hidden_states.cpu().numpy().tolist(),
                'labels': self._targets.cpu().numpy().tolist(),
            })
        return self._dataset

    def batches(
        self,
        batch_size: int,
        shuffle: bool = False,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        訓練用にバッチ化されたシーケンスを反復。

        Args:
            batch_size: バッチあたりのシーケンス数
            shuffle: バッチ化前にシャッフルするか

        Yields:
            (hidden_states, targets)のタプル
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
        訓練セットと検証セットに分割。

        Hugging Face Datasetのtrain_test_split機能を使用。

        Args:
            train_ratio: 訓練用データの割合

        Returns:
            (train_data, val_data)のタプル
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
    def empty(cls, seq_len: int, dim: int, device: Optional[str] = None) -> "SequenceData":
        """空のSequenceDataインスタンスを作成。"""
        hidden_states = torch.empty(0, seq_len, dim)
        targets = torch.empty(0, seq_len, dtype=torch.long)
        if device:
            hidden_states = hidden_states.to(device)
            targets = targets.to(device)
        return cls(hidden_states, targets)

    @classmethod
    def from_hf_dataset(cls, dataset: Dataset, device: Optional[str] = None) -> "SequenceData":
        """
        Hugging Face DatasetからSequenceDataを作成。

        Args:
            dataset: hidden_statesとlabelsを含むHugging Face Dataset
            device: 配置するデバイス

        Returns:
            SequenceDataインスタンス
        """
        hidden_states = torch.tensor(dataset['hidden_states'])
        targets = torch.tensor(dataset['labels'], dtype=torch.long)

        if device:
            hidden_states = hidden_states.to(device)
            targets = targets.to(device)

        return cls(hidden_states, targets)

    def select(self, indices: List[int]) -> "SequenceData":
        """
        指定インデックスのサブセットを選択。

        Args:
            indices: 選択するインデックスのリスト

        Returns:
            選択されたデータを含むSequenceData
        """
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        return SequenceData(
            self._hidden_states[indices_tensor],
            self._targets[indices_tensor]
        )
