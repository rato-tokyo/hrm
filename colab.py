"""
Google Colab用: Standard vs Deep Supervision Model比較実験

使い方:
!git clone https://github.com/rato-tokyo/hrm.git
%cd hrm
!python colab.py
"""

import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, 'src')
from ease import (
    DeepSupervisionTransformer,
    StandardTransformer,
    Trainer,
    create_deep_supervision_config,
    create_standard_config,
)


# ================================================================================
# 設定
# ================================================================================

@dataclass
class Config:
    """実験設定"""
    # モデル設定
    vocab_size: int = 1000
    seq_len: int = 32
    dim: int = 64
    num_layers: int = 3
    num_heads: int = 4
    base_lr: float = 1e-3
    pattern_length: int = 10

    # Phase 1: 小規模
    phase1_samples: int = 1000
    phase1_batch: int = 32
    phase1_epochs: int = 50
    phase1_patience: int = 1

    # Phase 2: 中規模
    phase2_samples: int = 10000
    phase2_batch: int = 64
    phase2_epochs: int = 50
    phase2_patience: int = 1


CONFIG = Config()


# ================================================================================
# データ生成
# ================================================================================

def load_wikitext_data(seq_len: int = 32, seed: int = 42):
    """WikiText-2データをロード"""
    try:
        from torchtext.datasets import WikiText2
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    except ImportError:
        print("torchtext not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'torchtext', 'portalocker'])
        from torchtext.datasets import WikiText2
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator

    torch.manual_seed(seed)

    # Tokenizerとデータセット
    tokenizer = get_tokenizer('basic_english')
    train_iter = WikiText2(split='train')

    # Vocabulary構築
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # データをトークン化
    def data_process(raw_text_iter):
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, val_iter = WikiText2(split=('train', 'valid'))
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)

    return train_data, val_data, len(vocab)


def batchify(data: torch.Tensor, batch_size: int, seq_len: int):
    """データをバッチ化"""
    # シーケンス数を計算
    num_seqs = data.size(0) // (seq_len + 1)
    data = data[:num_seqs * (seq_len + 1)]

    # (num_seqs, seq_len + 1)に整形
    data = data.view(num_seqs, seq_len + 1)

    # 入力とターゲットに分割
    x = data[:, :-1]
    y = data[:, 1:]

    # DataLoaderを作成
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_dataloaders(num_samples: int, batch_size: int, seq_len: int = 32, seed: int = 42) -> tuple:
    """WikiText-2 DataLoaderを作成"""
    train_data, val_data, vocab_size = load_wikitext_data(seq_len, seed)

    # vocab_sizeをCONFIGに保存
    CONFIG.vocab_size = vocab_size

    # サンプル数を制限（指定された数まで）
    max_tokens_train = num_samples * (seq_len + 1)
    max_tokens_val = int(num_samples * 0.2) * (seq_len + 1)

    train_data = train_data[:max_tokens_train]
    val_data = val_data[:max_tokens_val]

    train_loader = batchify(train_data, batch_size, seq_len)
    val_loader = batchify(val_data, batch_size, seq_len)

    print(f"Vocab size: {vocab_size}")

    return train_loader, val_loader


# ================================================================================
# 実験実行
# ================================================================================

def run_single_experiment(
    model_name: str,
    model_class: type,
    config_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    patience: int,
    device: str
) -> Dict:
    """単一モデルの実験を実行"""
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"Early Stopping: patience={patience}")
    print(f"{'='*60}")

    model = model_class(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.num_layers,
        num_heads=CONFIG.num_heads
    ).to(device)

    trainer_config = config_fn(num_layers=CONFIG.num_layers)
    trainer = Trainer(trainer_config, vocab_size=CONFIG.vocab_size, device=device)
    optimizer = trainer.create_optimizer(model, base_lr=CONFIG.base_lr)

    start = time.time()
    result = trainer.train_with_early_stopping(
        model, train_loader, val_loader, optimizer,
        max_epochs=max_epochs, patience=patience, verbose=True
    )

    return {
        'model_name': model_name,
        'train_losses': result['train_losses'],
        'val_ppls': result['val_losses'],
        'val_accs': [acc * 100 for acc in result['val_accs']],
        'best_acc': max(result['val_accs']) * 100,
        'best_epoch': result['best_epoch'],
        'time': time.time() - start,
        'stopped_early': result['stopped_early']
    }


def run_phase(phase_name: str, num_samples: int, batch_size: int, max_epochs: int, patience: int, device: str) -> List[Dict]:
    """1つのフェーズ（StandardとDeep Supervisionの両方）を実行"""
    print(f"\n{'='*60}")
    print(f"{phase_name}")
    print(f"{'='*60}")
    print(f"Samples: {num_samples}, Batch: {batch_size}")

    train_loader, val_loader = create_dataloaders(num_samples, batch_size, CONFIG.seq_len)

    results = []
    for model_name, model_class, config_fn in [
        ("Standard", StandardTransformer, create_standard_config),
        ("Deep Supervision", DeepSupervisionTransformer, create_deep_supervision_config)
    ]:
        result = run_single_experiment(
            model_name, model_class, config_fn,
            train_loader, val_loader, max_epochs, patience, device
        )
        results.append(result)

    return results


# ================================================================================
# 可視化
# ================================================================================

def plot_results(results: List[Dict], phase_name: str):
    """結果をプロット"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Train Loss
    for r in results:
        axes[0].plot(r['train_losses'], label=r['model_name'], marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title(f'{phase_name} - Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val PPL
    for r in results:
        axes[1].plot(r['val_ppls'], label=r['model_name'], marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Perplexity')
    axes[1].set_title(f'{phase_name} - Validation PPL')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Val Acc
    for r in results:
        axes[2].plot(r['val_accs'], label=r['model_name'], marker='o')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Val Accuracy (%)')
    axes[2].set_title(f'{phase_name} - Validation Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary
    print(f"\n{'='*70}")
    print(f"{phase_name} - SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Best Acc':<12} {'Time':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['model_name']:<25} {r['best_acc']:>10.2f}% {r['time']:>8.2f}s")
    print("="*70)


def print_final_comparison(phase1_results: List[Dict], phase2_results: List[Dict]):
    """最終比較を表示"""
    print("\n" + "="*90)
    print("FINAL COMPARISON")
    print("="*90)
    print(f"{'Phase':<20} {'Model':<25} {'Best Acc':<12} {'Time':<10}")
    print("-"*90)

    for r in phase1_results:
        print(f"{'Phase 1 (1K)':<20} {r['model_name']:<25} {r['best_acc']:>10.2f}% {r['time']:>8.2f}s")
    print("-"*90)
    for r in phase2_results:
        print(f"{'Phase 2 (10K)':<20} {r['model_name']:<25} {r['best_acc']:>10.2f}% {r['time']:>8.2f}s")
    print("="*90)

    # Winner
    p2_std = phase2_results[0]['best_acc']
    p2_deep = phase2_results[1]['best_acc']

    if p2_deep > p2_std:
        print(f"\nWINNER: Deep Supervision (+{p2_deep - p2_std:.2f}%)")
    elif p2_std > p2_deep:
        print(f"\nWINNER: Standard (+{p2_std - p2_deep:.2f}%)")
    else:
        print("\nWINNER: Tie")


# ================================================================================
# メイン
# ================================================================================

def main():
    """メイン実行関数"""
    print("="*60)
    print("EASE Framework: Standard vs Deep Supervision")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Phase 1
    phase1_results = run_phase(
        "Phase 1: Small-scale (1K)",
        CONFIG.phase1_samples,
        CONFIG.phase1_batch,
        CONFIG.phase1_epochs,
        CONFIG.phase1_patience,
        device
    )
    plot_results(phase1_results, "Phase 1")

    # Phase 2
    phase2_results = run_phase(
        "Phase 2: Medium-scale (10K)",
        CONFIG.phase2_samples,
        CONFIG.phase2_batch,
        CONFIG.phase2_epochs,
        CONFIG.phase2_patience,
        device
    )
    plot_results(phase2_results, "Phase 2")

    # Final comparison
    print_final_comparison(phase1_results, phase2_results)

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
