"""
CASCADEフレームワーク - 設定クラス

CascadeConfig: CASCADE固有の設定（patience, hard_ratio）
ExperimentConfig: モデルアーキテクチャと実験設定
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CascadeConfig:
    """
    CASCADE固有の設定。

    Hugging Face TrainingArgumentsに含まれない、
    CASCADEフレームワーク固有のパラメータを保持。

    Attributes:
        patience: Early stoppingの待機エポック数
        hard_ratio: hard tokenとして収集するトークンの割合 (0.0-1.0)
        lr_decay: 後続LLMごとの学習率減衰係数
    """
    patience: int
    hard_ratio: float
    lr_decay: float = 0.5


@dataclass
class ExperimentConfig:
    """
    CASCADE実験用の設定。

    モデルアーキテクチャとデータ設定。
    全フィールド必須（デフォルト値なし）。

    Attributes:
        dim: モデル次元
        num_heads: Attentionヘッド数
        ffn_dim: FFN隠れ層次元
        max_seq_len: 最大シーケンス長
        causal: Causalマスクを使用するか
        eps: 正規化のイプシロン
        seq_len: 言語モデリング用シーケンス長
        num_samples: 訓練サンプル数
        llm_layers: 各LLMのレイヤー数（例: (2, 2)は2層×2 LLM）
        tokenizer_name: 使用するトークナイザ名（デフォルト: "gpt2"）
    """
    dim: int
    num_heads: int
    ffn_dim: int
    max_seq_len: int
    causal: bool
    eps: float
    seq_len: int
    num_samples: int
    llm_layers: Tuple[int, ...]
    tokenizer_name: str = "gpt2"

    def to_gpt2_config(self, vocab_size: int):
        """
        GPT2Configに変換。

        Args:
            vocab_size: 語彙サイズ

        Returns:
            GPT2Configインスタンス
        """
        from transformers import GPT2Config
        return GPT2Config(
            vocab_size=vocab_size,
            n_embd=self.dim,
            n_head=self.num_heads,
            n_inner=self.ffn_dim,
            n_positions=self.max_seq_len,
            layer_norm_epsilon=self.eps,
        )
