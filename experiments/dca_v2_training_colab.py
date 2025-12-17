#!/usr/bin/env python3
"""
DCA-LLM v2 訓練・評価スクリプト (Colab用)

改善点:
1. 長系列（1024トークン）での検証 - DCAの本来の価値を発揮
2. DCAをGPT-2内部に統合 - 後付けではなく置換
3. 公平なパラメータ比較 - Baseline 5層 vs DCA内蔵4層

Colab実行:
    %cd /content
    !rm -rf hrm
    !git clone https://github.com/rato-tokyo/hrm.git
    !pip install transformers datasets
    !python /content/hrm/experiments/dca_v2_training_colab.py
"""

import sys
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

import torch

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from cascade.dca_llm import create_integrated_dca_llm
from cascade.dataloader import create_wikitext_dataloaders
from cascade.trainer_utils import (
    train_model,
    get_memory_usage,
    create_baseline_gpt2,
)


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

    # DCA固有の設定
    window_size: int = 256         # L0ウィンドウサイズ
    compression_ratio: int = 4     # L1圧縮率

    # 訓練設定
    batch_size: int = 8        # 長系列なのでバッチサイズ小さめ
    seq_len: int = 1024        # 長系列
    num_samples: int = 2500    # 訓練時間短縮のため半減
    num_epochs: int = 15
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Early stopping
    patience: int = 1

    # その他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

    results = {}

    # ===== Baseline (GPT-2 5層) =====
    print("\n" + "="*70)
    print(f"Training Baseline (GPT-2 {config.baseline_layers} layers)")
    print("="*70)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline_model = create_baseline_gpt2(
        vocab_size=vocab_size,
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.baseline_layers,
        max_seq_len=config.max_seq_len,
    )
    baseline_results = train_model(
        baseline_model,
        train_batches,
        val_batches,
        device=config.device,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        patience=config.patience,
        model_name=f"Baseline (GPT-2 {config.baseline_layers}L)",
    )
    baseline_results['memory'] = get_memory_usage()
    results['baseline'] = baseline_results

    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== IntegratedDCA (4層) =====
    print("\n" + "="*70)
    print(f"Training IntegratedDCA ({config.dca_layers} layers)")
    print("="*70)

    dca_model = create_integrated_dca_llm(
        vocab_size=vocab_size,
        dim=config.dim,
        num_layers=config.dca_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        window_size=config.window_size,
        compression_ratio=config.compression_ratio,
        device=config.device,
    )

    dca_results = train_model(
        dca_model,
        train_batches,
        val_batches,
        device=config.device,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        patience=config.patience,
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
