"""
CASCADEフレームワーク - 設定クラス

TrainerConfig: train_llm()用の訓練ハイパーパラメータ（TrainingArgumentsと互換）
ExperimentConfig: モデルアーキテクチャと実験設定
"""

from dataclasses import dataclass
from typing import Tuple
from transformers import TrainingArguments


@dataclass
class TrainerConfig:
    """
    LLM訓練用の設定。

    train_llm()で訓練ハイパーパラメータを設定するために使用。
    Hugging Face TrainingArgumentsと互換性を持つ。

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

    def to_training_arguments(
        self,
        output_dir: str = "./cascade_trainer_output",
    ) -> TrainingArguments:
        """
        Hugging Face TrainingArgumentsに変換。

        Args:
            output_dir: 出力ディレクトリ

        Returns:
            TrainingArgumentsインスタンス
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.max_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.lr,
            max_grad_norm=self.grad_clip,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=1 if self.verbose else 10,
            report_to="none",
            remove_unused_columns=False,
            disable_tqdm=not self.verbose,
        )

    @classmethod
    def from_training_arguments(
        cls,
        args: TrainingArguments,
        patience: int,
        hard_ratio: float,
        verbose: bool = True,
    ) -> "TrainerConfig":
        """
        Hugging Face TrainingArgumentsから作成。

        Args:
            args: TrainingArgumentsインスタンス
            patience: Early stoppingの待機エポック数
            hard_ratio: hard tokenとして収集するトークンの割合
            verbose: 訓練進捗を表示するか

        Returns:
            TrainerConfigインスタンス
        """
        return cls(
            batch_size=args.per_device_train_batch_size,
            max_epochs=int(args.num_train_epochs),
            patience=patience,
            grad_clip=args.max_grad_norm,
            hard_ratio=hard_ratio,
            lr=args.learning_rate,
            verbose=verbose,
        )


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
