"""
LEGOフレームワーク - SequenceData

LEGOブロック訓練用のデータコンテナ（シーケンスベース）。
"""

import torch
from typing import Iterator, Tuple, Optional


class SequenceData:
    """
    LEGOブロック訓練用データのコンテナ（シーケンスベース）。

    ブロック訓練用の(hidden_states, targets)シーケンスを保持。
    Attention計算のためにシーケンス構造を維持。

    Args:
        hidden_states: hidden statesテンソル (num_sequences, seq_len, dim)
        targets: ターゲットラベルテンソル (num_sequences, seq_len)

    使用例:
        # テンソルから作成
        data = SequenceData(hidden_states, targets)

        # バッチで反復（シーケンスを維持）
        for h, y in data.batches(batch_size=8):
            # h: (batch_size, seq_len, dim)
            # y: (batch_size, seq_len)
            ...

        # train/valに分割
        train_data, val_data = data.split(train_ratio=0.8)
    """

    def __init__(self, hidden_states: torch.Tensor, targets: torch.Tensor):
        if hidden_states.shape[0] != targets.shape[0]:
            raise ValueError(
                f"バッチサイズ不一致: hidden_states={hidden_states.shape[0]}, targets={targets.shape[0]}"
            )
        if hidden_states.shape[1] != targets.shape[1]:
            raise ValueError(
                f"シーケンス長不一致: hidden_states={hidden_states.shape[1]}, targets={targets.shape[1]}"
            )
        self._hidden_states = hidden_states  # (num_sequences, seq_len, dim)
        self._targets = targets  # (num_sequences, seq_len)

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
        return self._hidden_states.shape[1]

    @property
    def num_sequences(self) -> int:
        """シーケンス数。"""
        return self._hidden_states.shape[0]

    @property
    def num_tokens(self) -> int:
        """総トークン数。"""
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

    def batches(
        self,
        batch_size: int,
        shuffle: bool
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
    def empty(cls, seq_len: int, dim: int, device: Optional[str]) -> "SequenceData":
        """空のSequenceDataインスタンスを作成。"""
        hidden_states = torch.empty(0, seq_len, dim)
        targets = torch.empty(0, seq_len, dtype=torch.long)
        if device:
            hidden_states = hidden_states.to(device)
            targets = targets.to(device)
        return cls(hidden_states, targets)
