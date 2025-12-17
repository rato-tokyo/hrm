#!/usr/bin/env python3
"""
DCA-LLM v2 訓練・評価スクリプト

改善点:
1. 長系列（1024トークン）での検証 - DCAの本来の価値を発揮
2. DCAをGPT-2内部に統合 - 後付けではなく置換
3. 公平なパラメータ比較 - Baseline 5層 vs DCA内蔵4層

出力:
- Train PPL / Val PPL
- ベースラインとの比較
- パラメータ数の比較
- 訓練時間

使用方法:
    python experiments/dca_v2_training.py
"""

import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from cascade.dca_llm import create_integrated_dca_llm
from cascade.dataloader import create_wikitext_dataloaders
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass
class TrainingConfig:
    """訓練設定。"""
    # モデル設定
    dim: int = 256
    num_heads: int = 4
    max_seq_len: int = 1024  # 長系列

    # レイヤー数（公平な比較のため）
    baseline_layers: int = 5   # Baseline: 5層
    dca_layers: int = 4        # DCA: 4層（パラメータ数を揃える）

    # 訓練設定
    batch_size: int = 8        # 長系列なのでバッチサイズ小さめ
    seq_len: int = 1024        # 長系列
    num_samples: int = 5000
    num_epochs: int = 15
    learning_rate: float = 2.5e-4
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

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

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
    best_state = None

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_batches, optimizer, device, config.max_grad_norm)
        train_ppl = math.exp(train_loss)
        val_ppl = compute_ppl(model, val_batches, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)

        best_marker = ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_marker = " <- Best"
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train PPL: {train_ppl:8.2f} | Val PPL: {val_ppl:8.2f} | "
              f"Time: {epoch_time:.1f}s{best_marker}")

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

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


def create_baseline_gpt2(vocab_size: int, config: TrainingConfig) -> nn.Module:
    """Baseline GPT-2モデルを作成（5層）。"""
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=config.dim,
        n_head=config.num_heads,
        n_layer=config.baseline_layers,  # 5層
        n_positions=config.max_seq_len,
        loss_type="ForCausalLMLoss",  # 明示的に設定して警告を抑制
    )
    return GPT2LMHeadModel(gpt2_config)


def run_experiment(config: TrainingConfig) -> Dict:
    """実験を実行。"""
    print("="*70)
    print("DCA-LLM v2 Training Experiment")
    print("="*70)
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*70)
    print("\nExperiment Settings:")
    print(f"  Sequence Length: {config.seq_len} (long sequence)")
    print(f"  Baseline: GPT-2 with {config.baseline_layers} layers")
    print(f"  DCA: IntegratedDCA with {config.dca_layers} layers")
    print("="*70)

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

    # ===== Baseline (GPT-2 5層) =====
    print("\n" + "="*70)
    print(f"Training Baseline (GPT-2 {config.baseline_layers} layers)")
    print("="*70)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    baseline_model = create_baseline_gpt2(vocab_size, config)
    baseline_results = train_model(
        baseline_model,
        train_batches,
        val_batches,
        config,
        model_name=f"Baseline (GPT-2 {config.baseline_layers}L)",
    )
    baseline_results['memory'] = get_memory_usage()
    results['baseline'] = baseline_results

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== IntegratedDCA (4層) =====
    print("\n" + "="*70)
    print(f"Training IntegratedDCA ({config.dca_layers} layers)")
    print("="*70)

    dca_model = create_integrated_dca_llm(
        vocab_size=vocab_size,
        dim=config.dim,
        num_layers=config.dca_layers,  # 4層
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        device=config.device,
    )

    dca_results = train_model(
        dca_model,
        train_batches,
        val_batches,
        config,
        model_name=f"IntegratedDCA ({config.dca_layers}L)",
    )
    dca_results['memory'] = get_memory_usage()
    results['dca'] = dca_results

    # ===== 結果サマリ =====
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Model':<30} {'Train PPL':>12} {'Val PPL':>12} {'Params':>12} {'Time':>10}")
    print("-"*70)

    for name, res in results.items():
        print(f"{res['model_name']:<30} "
              f"{res['train_ppl']:>12.2f} "
              f"{res['val_ppl']:>12.2f} "
              f"{res['num_params']:>12,} "
              f"{res['total_time']:>9.1f}s")

    print("-"*70)

    baseline_val = results['baseline']['val_ppl']
    dca_val = results['dca']['val_ppl']
    baseline_params = results['baseline']['num_params']
    dca_params = results['dca']['num_params']

    diff = dca_val - baseline_val
    diff_pct = (diff / baseline_val) * 100
    param_diff = dca_params - baseline_params
    param_diff_pct = (param_diff / baseline_params) * 100

    print("\nDCA vs Baseline:")
    print(f"  Val PPL difference: {diff:+.2f} ({diff_pct:+.1f}%)")
    print(f"  Parameter difference: {param_diff:+,} ({param_diff_pct:+.1f}%)")

    if diff < 0:
        print(f"  >>> DCA improves perplexity by {abs(diff):.2f} ({abs(diff_pct):.1f}%)")
    else:
        print(f"  >>> Baseline is better by {diff:.2f} ({diff_pct:.1f}%)")

    return results


if __name__ == "__main__":
    config = TrainingConfig()
    run_experiment(config)
