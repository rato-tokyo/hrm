"""
LEGOフレームワーク - 設定クラス

TrainerConfig: train_llm()用の訓練ハイパーパラメータ
ExperimentConfig: モデルアーキテクチャと実験設定
"""

from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """
    LLM訓練用の設定。

    train_llm()で訓練ハイパーパラメータを設定するために使用。
    全フィールド必須（デフォルト値なし）。

    Exit判定はexit_fnとhidden_historyを使用（デフォルト: CALM式cos_sim）。

    Attributes:
        batch_size: 訓練バッチサイズ
        max_epochs: 最大エポック数
        patience: Early stoppingの待機エポック数
        grad_clip: 勾配クリッピング値
        hard_ratio: hard tokenとして収集するトークンの割合
        lr: 学習率
        verbose: 訓練進捗を表示するか
    """
    batch_size: int
    max_epochs: int
    patience: int
    grad_clip: float
    hard_ratio: float
    lr: float
    verbose: bool


@dataclass
class ExperimentConfig:
    """
    LEGO実験用の設定。

    モデルアーキテクチャとデータ設定。
    全フィールド必須（デフォルト値なし）。

    Attributes:
        dim: モデル次元
        num_heads: Attentionヘッド数
        ffn_dim: FFN隠れ層次元
        max_seq_len: 最大シーケンス長
        causal: Causalマスクを使用するか
        eps: RMSNormのイプシロン
        seq_len: 言語モデリング用シーケンス長
        num_samples: 訓練サンプル数
        llm_layers: 各LLMのレイヤー数（例: (2, 2)は2層×2 LLM）
    """
    dim: int
    num_heads: int
    ffn_dim: int
    max_seq_len: int
    causal: bool
    eps: float
    seq_len: int
    num_samples: int
    llm_layers: tuple[int, ...]
