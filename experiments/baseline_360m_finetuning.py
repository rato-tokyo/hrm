"""
ベースライン実験: SmolLM2-360Mファインチューニング

CASCADE実験と比較するためのベースライン。
同じデータセット、同じ訓練設定でSmolLM2-360Mをファインチューニングする。

Colabでの実行方法:
    !python experiments/baseline_360m_finetuning.py
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 実験パラメータ（ここを編集して実験設定を変更）
# ============================================================

# モデル
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

# 訓練設定
EPOCHS = 10               # 最大エポック数
BATCH_SIZE = 16           # バッチサイズ（360Mは大きいため小さめに）
LEARNING_RATE = 1e-5      # 学習率（ファインチューニングなので小さめ）
SEQ_LEN = 128             # シーケンス長
PATIENCE = 3              # Early stoppingのpatience

# データ設定（CASCADE実験と同じにする）
NUM_TRAIN_SAMPLES = 1000  # 訓練サンプル数
NUM_VAL_SAMPLES = 100     # 検証サンプル数

# その他
SEED = 42                 # 乱数シード
OUTPUT_DIR = None         # 出力ディレクトリ（Noneで自動生成）

# ============================================================
# 以下は実装コード（通常は編集不要）
# ============================================================


@dataclass
class TrainingResult:
    """訓練結果"""
    model_name: str
    total_params: int
    train_tokens: int
    val_tokens: int
    final_train_loss: float
    final_val_loss: float
    val_ppl: float
    best_val_ppl: float
    total_epochs: int
    training_time: float
    model_path: str


@dataclass
class ExperimentConfig:
    """実験設定"""
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    seq_len: int
    patience: int
    num_train_samples: int
    num_val_samples: int
    seed: int
    output_dir: str


def set_seed(seed: int):
    """乱数シードを設定"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloaders(
    tokenizer,
    num_train: int,
    num_val: int,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Alpacaデータセットからデータローダーを作成"""
    from datasets import load_dataset

    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    def tokenize_and_chunk(max_samples: int) -> List[List[int]]:
        all_input_ids: List[List[int]] = []

        for i in range(len(dataset)):
            if len(all_input_ids) >= max_samples:
                break

            full_text = f"{dataset[i]['instruction']}\n{dataset[i]['input']}\n{dataset[i]['output']}"
            tokens = tokenizer.encode(full_text, add_special_tokens=True)

            for j in range(0, len(tokens) - seq_len, seq_len):
                if len(all_input_ids) >= max_samples:
                    break
                chunk = tokens[j:j + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    all_input_ids.append(chunk)

        return all_input_ids

    print("  Alpacaデータセットをトークナイズ中...")
    all_data = tokenize_and_chunk(num_train + num_val)
    print(f"  取得サンプル数: {len(all_data)}")

    train_data = torch.tensor(all_data[:num_train], dtype=torch.long)
    val_data = torch.tensor(all_data[num_train:num_train + num_val], dtype=torch.long)

    train_x, train_y = train_data[:, :-1], train_data[:, 1:]
    val_x, val_y = val_data[:, :-1], val_data[:, 1:]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
) -> Tuple[float, float, float, float, int]:
    """モデルを訓練"""
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    patience_counter = 0
    actual_epochs = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        actual_epochs = epoch + 1

        # 訓練
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(x_batch, return_dict=True)
            logits = outputs.logits

            # (batch, seq, vocab) -> (batch * seq, vocab)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1)
            )
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y_batch.numel()
            train_count += y_batch.numel()

        train_loss = train_loss_sum / train_count

        # 検証
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(x_batch, return_dict=True)
                logits = outputs.logits

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y_batch.view(-1)
                )
                val_loss_sum += loss.item() * y_batch.numel()
                val_count += y_batch.numel()

        val_loss = val_loss_sum / val_count
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        epoch_time = time.time() - epoch_start

        print(f"  Epoch {epoch + 1}/{epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_ppl={val_ppl:.2f}, time={epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                break

    return train_loss, val_loss, val_ppl, best_val_ppl, actual_epochs


def save_model(model, tokenizer, output_dir: Path):
    """モデルを保存"""
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    return str(model_dir)


def run_experiment():
    """実験を実行"""
    # 出力ディレクトリを設定
    output_dir_str = OUTPUT_DIR
    if output_dir_str is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_str = f"outputs/baseline_360m_{timestamp}"

    config = ExperimentConfig(
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seq_len=SEQ_LEN,
        patience=PATIENCE,
        num_train_samples=NUM_TRAIN_SAMPLES,
        num_val_samples=NUM_VAL_SAMPLES,
        seed=SEED,
        output_dir=output_dir_str,
    )

    print("=" * 80)
    print("ベースライン実験: SmolLM2-360Mファインチューニング")
    print("=" * 80)
    print("\n設定:")
    print(f"  モデル: {config.model_name}")
    print(f"  エポック数: {config.epochs}")
    print(f"  バッチサイズ: {config.batch_size}")
    print(f"  学習率: {config.learning_rate}")
    print(f"  Early stopping patience: {config.patience}")
    print()

    device = get_device()
    print(f"デバイス: {device}")
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 設定を保存
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    # モデルをロード
    print(f"\nモデルをロード中: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    total_params = sum(p.numel() for p in model.parameters())
    num_layers = model.config.num_hidden_layers
    print(f"  パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  レイヤー数: {num_layers}")

    # データをロード
    print("\nデータをロード中...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        config.num_train_samples,
        config.num_val_samples,
        config.seq_len,
        config.batch_size,
    )

    train_tokens = len(train_loader.dataset) * config.seq_len
    val_tokens = len(val_loader.dataset) * config.seq_len
    print(f"  訓練サンプル数: {len(train_loader.dataset)}")
    print(f"  検証サンプル数: {len(val_loader.dataset)}")
    print(f"  訓練トークン数: {train_tokens:,}")
    print(f"  検証トークン数: {val_tokens:,}")

    # 訓練
    print("\n訓練開始...")
    start_time = time.time()

    train_loss, val_loss, val_ppl, best_val_ppl, actual_epochs = train_model(
        model,
        train_loader,
        val_loader,
        config.epochs,
        config.learning_rate,
        config.patience,
        device,
    )

    training_time = time.time() - start_time
    print(f"\n訓練完了: {training_time:.1f}秒")
    print(f"  最終val_loss: {val_loss:.4f}")
    print(f"  最終val_ppl: {val_ppl:.2f}")
    print(f"  最良val_ppl: {best_val_ppl:.2f}")

    # モデルを保存
    model_path = save_model(model, tokenizer, output_dir)
    print(f"  モデル保存先: {model_path}")

    # 結果を保存
    result = TrainingResult(
        model_name=config.model_name,
        total_params=total_params,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        val_ppl=val_ppl,
        best_val_ppl=best_val_ppl,
        total_epochs=actual_epochs,
        training_time=training_time,
        model_path=model_path,
    )

    with open(output_dir / "result.json", "w") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    # サマリー
    print(f"\n{'=' * 80}")
    print("実験完了")
    print(f"{'=' * 80}")
    print("\n【結果サマリー】")
    print(f"  モデル: {result.model_name}")
    print(f"  パラメータ数: {result.total_params/1e6:.1f}M")
    print(f"  訓練トークン数: {result.train_tokens:,}")
    print(f"  訓練エポック数: {result.total_epochs}")
    print(f"  訓練時間: {result.training_time:.1f}秒")
    print(f"  最良val_ppl: {result.best_val_ppl:.2f}")
    print(f"\n結果保存先: {output_dir}")

    return result


if __name__ == "__main__":
    run_experiment()
