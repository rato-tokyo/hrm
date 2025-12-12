"""
Google Colab用: Standard vs Deep Supervision Model比較実験

使い方:
1. Google Colabで新しいノートブックを作成
2. GPU有効化: Runtime → Change runtime type → T4 GPU or L4 GPU
3. 以下のコマンドを実行:

!git clone https://github.com/rato-tokyo/hrm.git
%cd hrm
!python colab.py
"""

import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# EASEフレームワークのインポート
sys.path.insert(0, 'src')
from ease import (
    StandardTransformer,
    DeepSupervisionTransformer,
    Trainer,
    TrainingConfig,
    create_standard_config,
    create_deep_supervision_config
)


# ================================================================================
# データ準備
# ================================================================================

def create_synthetic_data(num_samples: int, vocab_size: int, seq_len: int, seed: int = 42):
    """シンプルな合成データを作成（次トークン予測タスク）"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.cat([x[:, 1:], torch.randint(0, vocab_size, (num_samples, 1))], dim=1)

    return x, y


def prepare_dataloaders(num_samples: int, vocab_size: int = 1000, seq_len: int = 32,
                       batch_size: int = 32, seed: int = 42):
    """DataLoaderを準備"""
    x, y = create_synthetic_data(num_samples, vocab_size, seq_len, seed)

    # Train/Val分割 (80/20)
    split_idx = int(num_samples * 0.8)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("\nDataset prepared:")
    print(f"  Train samples: {len(x_train)}")
    print(f"  Val samples: {len(x_val)}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader


# ================================================================================
# 実験実行
# ================================================================================

def run_experiment(model_name: str, model: nn.Module, config: TrainingConfig,
                  train_loader: DataLoader, val_loader: DataLoader,
                  num_epochs: int = 10, base_lr: float = 1e-3,
                  device: str = 'cuda', use_early_stopping: bool = False,
                  patience: int = 1) -> Dict:
    """単一モデルの実験を実行"""
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name}")
    if use_early_stopping:
        print(f"Early Stopping: enabled (patience={patience})")
    print(f"{'='*60}")

    model = model.to(device)
    trainer = Trainer(config, vocab_size=1000, device=device)
    optimizer = trainer.create_optimizer(model, base_lr=base_lr)

    start_time = time.time()

    if use_early_stopping:
        # Early Stopping使用
        result = trainer.train_with_early_stopping(
            model=model,
            train_batches=train_loader,
            val_batches=val_loader,
            optimizer=optimizer,
            max_epochs=num_epochs,
            patience=patience,
            verbose=True
        )
        train_losses = result['train_losses']
        val_losses = result['val_losses']
        val_accs = [acc * 100 for acc in result['val_accs']]  # Convert to percentage
        stopped_early = result['stopped_early']
        best_epoch = result['best_epoch']
    else:
        # 通常訓練
        train_losses = []
        val_losses = []
        val_accs = []

        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch(model, train_loader, optimizer)
            train_losses.append(train_loss)

            val_stats = trainer.evaluate(model, val_loader)
            val_ppl = val_stats['ppl']
            val_acc = val_stats['acc'] * 100  # Convert to percentage

            val_losses.append(val_ppl)
            val_accs.append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val PPL: {val_ppl:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

        stopped_early = False
        best_epoch = val_accs.index(max(val_accs))

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f}s")
    print(f"Best Val Acc: {max(val_accs):.2f}% (epoch {best_epoch+1})")
    if use_early_stopping and stopped_early:
        print("Early stopping was triggered")

    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'training_time': training_time,
        'best_val_acc': max(val_accs),
        'best_epoch': best_epoch,
        'stopped_early': stopped_early if use_early_stopping else False
    }


# ================================================================================
# 結果可視化
# ================================================================================

def plot_comparison(results_list: List[Dict], phase_name: str):
    """複数モデルの結果を比較プロット"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Train Loss
    ax = axes[0]
    for result in results_list:
        ax.plot(result['train_losses'], label=result['model_name'], marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title(f'{phase_name} - Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val Perplexity
    ax = axes[1]
    for result in results_list:
        ax.plot(result['val_losses'], label=result['model_name'], marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Perplexity')
    ax.set_title(f'{phase_name} - Validation Perplexity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val Accuracy
    ax = axes[2]
    for result in results_list:
        ax.plot(result['val_accs'], label=result['model_name'], marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title(f'{phase_name} - Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary Table
    print(f"\n{'='*70}")
    print(f"{phase_name} - SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Best Val Acc':<15} {'Training Time':<15}")
    print("-"*70)
    for result in results_list:
        print(f"{result['model_name']:<30} {result['best_val_acc']:>12.2f}% {result['training_time']:>12.2f}s")
    print("="*70)


# ================================================================================
# メイン実行
# ================================================================================

def main():
    """メイン実験実行関数"""

    # GPU確認
    print("="*60)
    print("EASE Framework: Standard vs Deep Supervision Comparison")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ハイパーパラメータ
    VOCAB_SIZE = 1000
    SEQ_LEN = 32
    DIM = 64
    NUM_LAYERS = 3
    NUM_HEADS = 4
    BASE_LR = 1e-3

    # ===========================================================================
    # Phase 1: 小規模データで動作確認
    # ===========================================================================

    print("\n" + "="*60)
    print("Phase 1: Small-scale Data (Quick Test)")
    print("="*60)

    NUM_SAMPLES_SMALL = 1000
    BATCH_SIZE_SMALL = 32
    NUM_EPOCHS_SMALL = 20  # Early Stoppingで自動終了
    PATIENCE_SMALL = 1

    train_loader_small, val_loader_small = prepare_dataloaders(
        num_samples=NUM_SAMPLES_SMALL,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE_SMALL,
        seed=42
    )

    # Standard Model
    standard_model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS
    )
    standard_config = create_standard_config(num_layers=NUM_LAYERS)

    result_standard_small = run_experiment(
        model_name="Standard Model",
        model=standard_model,
        config=standard_config,
        train_loader=train_loader_small,
        val_loader=val_loader_small,
        num_epochs=NUM_EPOCHS_SMALL,
        base_lr=BASE_LR,
        device=device,
        use_early_stopping=True,
        patience=PATIENCE_SMALL
    )

    # Deep Supervision Model
    deep_supervision_model = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        exit_layer=1,
        routing_threshold=0.0
    )
    deep_supervision_config = create_deep_supervision_config(num_layers=NUM_LAYERS)

    result_deep_supervision_small = run_experiment(
        model_name="Deep Supervision Model",
        model=deep_supervision_model,
        config=deep_supervision_config,
        train_loader=train_loader_small,
        val_loader=val_loader_small,
        num_epochs=NUM_EPOCHS_SMALL,
        base_lr=BASE_LR,
        device=device,
        use_early_stopping=True,
        patience=PATIENCE_SMALL
    )

    # Phase 1結果可視化
    plot_comparison([result_standard_small, result_deep_supervision_small], "Phase 1")

    # ===========================================================================
    # Phase 2: 中規模データで精度比較
    # ===========================================================================

    print("\n" + "="*60)
    print("Phase 2: Medium-scale Data (Full Comparison)")
    print("="*60)

    NUM_SAMPLES_MEDIUM = 10000
    BATCH_SIZE_MEDIUM = 64
    NUM_EPOCHS_MEDIUM = 50  # Early Stoppingで自動終了
    PATIENCE_MEDIUM = 1

    train_loader_medium, val_loader_medium = prepare_dataloaders(
        num_samples=NUM_SAMPLES_MEDIUM,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE_MEDIUM,
        seed=42
    )

    # Standard Model
    standard_model_medium = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS
    )

    result_standard_medium = run_experiment(
        model_name="Standard Model",
        model=standard_model_medium,
        config=standard_config,
        train_loader=train_loader_medium,
        val_loader=val_loader_medium,
        num_epochs=NUM_EPOCHS_MEDIUM,
        base_lr=BASE_LR,
        device=device,
        use_early_stopping=True,
        patience=PATIENCE_MEDIUM
    )

    # Deep Supervision Model
    deep_supervision_model_medium = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        exit_layer=1,
        routing_threshold=0.0
    )

    result_deep_supervision_medium = run_experiment(
        model_name="Deep Supervision Model",
        model=deep_supervision_model_medium,
        config=deep_supervision_config,
        train_loader=train_loader_medium,
        val_loader=val_loader_medium,
        num_epochs=NUM_EPOCHS_MEDIUM,
        base_lr=BASE_LR,
        device=device,
        use_early_stopping=True,
        patience=PATIENCE_MEDIUM
    )

    # Phase 2結果可視化
    plot_comparison([result_standard_medium, result_deep_supervision_medium], "Phase 2")

    # ===========================================================================
    # 最終比較
    # ===========================================================================

    print("\n" + "="*90)
    print("FINAL COMPARISON: Standard vs Deep Supervision")
    print("="*90)
    print(f"{'Phase':<20} {'Model':<30} {'Best Val Acc':<15} {'Training Time':<15}")
    print("-"*90)

    print(f"{'Phase 1 (1K)':<20} {'Standard':<30} {result_standard_small['best_val_acc']:>12.2f}% {result_standard_small['training_time']:>12.2f}s")
    print(f"{'Phase 1 (1K)':<20} {'Deep Supervision':<30} {result_deep_supervision_small['best_val_acc']:>12.2f}% {result_deep_supervision_small['training_time']:>12.2f}s")
    print("-"*90)
    print(f"{'Phase 2 (10K)':<20} {'Standard':<30} {result_standard_medium['best_val_acc']:>12.2f}% {result_standard_medium['training_time']:>12.2f}s")
    print(f"{'Phase 2 (10K)':<20} {'Deep Supervision':<30} {result_deep_supervision_medium['best_val_acc']:>12.2f}% {result_deep_supervision_medium['training_time']:>12.2f}s")
    print("="*90)

    # 勝者判定
    if result_deep_supervision_medium['best_val_acc'] > result_standard_medium['best_val_acc']:
        winner = "Deep Supervision Model"
        improvement = result_deep_supervision_medium['best_val_acc'] - result_standard_medium['best_val_acc']
    elif result_standard_medium['best_val_acc'] > result_deep_supervision_medium['best_val_acc']:
        winner = "Standard Model"
        improvement = result_standard_medium['best_val_acc'] - result_deep_supervision_medium['best_val_acc']
    else:
        winner = "Tie"
        improvement = 0.0

    print(f"\nWINNER: {winner}")
    if winner != "Tie":
        print(f"Improvement: +{improvement:.2f}% accuracy")

    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
