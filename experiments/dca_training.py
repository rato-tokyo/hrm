#!/usr/bin/env python3
"""
DCA-LLM 訓練・評価スクリプト

DCA（Dual-Context Attention）とベースラインの比較実験を行う。

出力:
- Train PPL / Val PPL
- ベースラインとの比較
- メモリ使用量
- 訓練時間

使用方法:
    python experiments/dca_training.py
"""

import sys
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from cascade.dca_llm import create_dca_llm_from_scratch
from cascade.dataloader import create_wikitext_dataloaders


@dataclass
class TrainingConfig:
    """訓練設定。"""
    # モデル設定
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 512

    # DCA設定
    window_size: int = 128
    max_representatives: int = 64
    encoder_mode: str = "pooling"

    # 訓練設定
    batch_size: int = 16
    seq_len: int = 128
    num_samples: int = 5000
    num_epochs: int = 10
    learning_rate: float = 2.5e-4  # 1e-3 / 4 = 2.5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Early stopping
    patience: int = 1

    # その他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def compute_ppl(model: nn.Module, batches: List[Tuple[torch.Tensor, torch.Tensor]], device: str) -> float:
    """Perplexityを計算。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)

            # DCAモデルの場合はリセット
            if hasattr(model, 'reset_memory'):
                model.reset_memory()

            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss

            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def train_epoch(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float,
) -> float:
    """1エポック訓練。"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in batches:
        x, y = x.to(device), y.to(device)

        # DCAモデルの場合はリセット
        if hasattr(model, 'reset_memory'):
            model.reset_memory()

        optimizer.zero_grad()
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    return avg_loss


def train_model(
    model: nn.Module,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    config: TrainingConfig,
    model_name: str = "model",
) -> Dict:
    """モデルを訓練。"""
    device = config.device
    model = model.to(device)

    # オプティマイザとスケジューラ
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # 訓練ループ
    best_val_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0
    history: dict[str, list[float]] = {
        'train_loss': [],
        'train_ppl': [],
        'val_ppl': [],
    }

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # 訓練
        train_loss = train_epoch(model, train_batches, optimizer, device, config.max_grad_norm)
        train_ppl = math.exp(train_loss)

        # 検証
        val_ppl = compute_ppl(model, val_batches, device)

        # スケジューラ更新
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 履歴記録
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)

        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:8.2f} | "
              f"Val PPL: {val_ppl:8.2f} | Time: {epoch_time:.1f}s")

        # Early stopping
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch + 1
            patience_counter = 0
            # ベストモデルを保存（メモリ上）
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={config.patience})")
                break

    total_time = time.time() - start_time

    # ベストモデルを復元
    model.load_state_dict(best_state)
    model = model.to(device)

    # 最終評価
    final_train_ppl = compute_ppl(model, train_batches, device)
    final_val_ppl = compute_ppl(model, val_batches, device)

    print(f"\n{'='*60}")
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best epoch: {best_epoch}")
    print(f"Final Train PPL: {final_train_ppl:.2f}")
    print(f"Final Val PPL: {final_val_ppl:.2f}")
    print(f"{'='*60}")

    return {
        'model_name': model_name,
        'train_ppl': final_train_ppl,
        'val_ppl': final_val_ppl,
        'best_epoch': best_epoch,
        'total_time': total_time,
        'history': history,
        'num_params': sum(p.numel() for p in model.parameters()),
    }


def get_memory_usage() -> Dict[str, float]:
    """GPUメモリ使用量を取得。"""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        }
    return {'allocated_mb': 0, 'reserved_mb': 0}


def run_experiment(config: TrainingConfig) -> Dict:
    """実験を実行。"""
    print("="*70)
    print("DCA-LLM Training Experiment")
    print("="*70)
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*70)

    # シード設定
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # データ準備
    print("\nLoading WikiText-2 dataset...")
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        seed=config.seed,
    )
    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")
    print(f"Vocab size: {vocab_size}")

    results = {}

    # ===== Baseline (DCAなし) =====
    print("\n" + "="*70)
    print("Training Baseline (no DCA)")
    print("="*70)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    baseline_model = create_dca_llm_from_scratch(
        vocab_size=vocab_size,
        dim=config.dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        window_size=config.window_size,
        max_representatives=config.max_representatives,
        encoder_mode=config.encoder_mode,
        use_dca=False,  # DCAなし
        device=config.device,
    )

    baseline_results = train_model(
        baseline_model,
        train_batches,
        val_batches,
        config,
        model_name="Baseline (no DCA)",
    )
    baseline_results['memory'] = get_memory_usage()
    results['baseline'] = baseline_results

    # メモリ解放
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== DCA Model =====
    print("\n" + "="*70)
    print("Training DCA-LLM")
    print("="*70)

    dca_model = create_dca_llm_from_scratch(
        vocab_size=vocab_size,
        dim=config.dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        window_size=config.window_size,
        max_representatives=config.max_representatives,
        encoder_mode=config.encoder_mode,
        use_dca=True,  # DCAあり
        device=config.device,
    )

    dca_results = train_model(
        dca_model,
        train_batches,
        val_batches,
        config,
        model_name="DCA-LLM",
    )
    dca_results['memory'] = get_memory_usage()
    results['dca'] = dca_results

    # ===== 結果サマリ =====
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Train PPL':>12} {'Val PPL':>12} {'Params':>12} {'Time':>10}")
    print("-"*70)

    for name, res in results.items():
        print(f"{res['model_name']:<25} "
              f"{res['train_ppl']:>12.2f} "
              f"{res['val_ppl']:>12.2f} "
              f"{res['num_params']:>12,} "
              f"{res['total_time']:>9.1f}s")

    print("-"*70)

    # 比較
    baseline_val = results['baseline']['val_ppl']
    dca_val = results['dca']['val_ppl']
    diff = dca_val - baseline_val
    diff_pct = (diff / baseline_val) * 100

    print("\nDCA vs Baseline:")
    print(f"  Val PPL difference: {diff:+.2f} ({diff_pct:+.1f}%)")
    if diff < 0:
        print(f"  DCA improves perplexity by {abs(diff):.2f} ({abs(diff_pct):.1f}%)")
    else:
        print(f"  Baseline is better by {diff:.2f} ({diff_pct:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="DCA-LLM Training")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=1, help="Early stopping patience")
    parser.add_argument("--window_size", type=int, default=128, help="DCA window size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = TrainingConfig(
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        patience=args.patience,
        window_size=args.window_size,
        seed=args.seed,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
