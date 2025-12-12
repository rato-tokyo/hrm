#!/usr/bin/env python3
"""
EASE: Efficient Asymmetric Supervision for Early-Exit Transformers

Run experiments with different training configurations.

Usage:
    python run_experiments.py

See src/ease/ for the framework implementation.
See docs/experiments/ for detailed results and analysis.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from datetime import datetime
from typing import Dict, List

from ease import (
    DeepSupervisionTransformer,
    TrainingConfig,
    Trainer,
    create_standard_config,
    create_deep_supervision_config,
)
from experiments import set_seed, prepare_wikitext_data, ExperimentConfig


def run_experiment(
    name: str,
    config: TrainingConfig,
    model: DeepSupervisionTransformer,
    trainer: Trainer,
    train_batches: List,
    val_batches: List,
    lr: float = 1e-3,
    max_epochs: int = 50,
    grad_clip: float = 1.0,
    verbose: bool = True
) -> Dict:
    """Run experiment with trainer."""
    optimizer = trainer.create_optimizer(model, base_lr=lr)

    best_ppl = float('inf')
    best_epoch = 0
    best_stats: Dict = {}

    for epoch in range(max_epochs):
        train_loss = trainer.train_epoch(model, train_batches, optimizer, grad_clip)
        train_ppl = np.exp(train_loss)

        stats = trainer.evaluate(model, val_batches)
        val_ppl = stats['ppl']

        if verbose:
            routing_info = f", Shallow={stats['shallow_ratio']*100:.1f}%, Compute={stats['compute_cost']*100:.1f}%" if config.has_routing else ""
            print(f"  Epoch {epoch+1}: Train={train_ppl:.2f}, Val={val_ppl:.2f}{routing_info}")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            best_stats = stats.copy()
        else:
            if verbose:
                print("  >>> Early stopping")
            break

    result: Dict = {
        'name': name,
        'config': config.describe(),
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        **best_stats
    }

    return result


def main() -> None:
    print("=" * 70)
    print("EASE EXPERIMENTS - SIMPLIFIED FRAMEWORK")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    exp_config = ExperimentConfig(
        train_chars=100000,
        val_chars=10000,
        seq_len=64,
        batch_size=32,
        dim=64,
        num_heads=4,
        num_layers=3,
        lr=1e-3,
        max_epochs=50,
        patience=0,
        device='cpu',
        seed=42,
    )

    print(f"\nConfig: {exp_config.train_chars:,} train chars, dim={exp_config.dim}, layers={exp_config.num_layers}")
    print("Early stopping: Stop immediately when val PPL worsens\n")

    # Prepare data
    train_batches, val_batches, vocab_size, _ = prepare_wikitext_data(
        exp_config.train_chars, exp_config.val_chars,
        exp_config.seq_len, exp_config.batch_size
    )
    print(f"Data: {len(train_batches)} train batches, {len(val_batches)} val batches")
    print(f"Vocab size: {vocab_size}")

    results: List[Dict] = []

    # =========================================================================
    # Part 1: Base Models (No Options)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Base Models")
    print("=" * 70)

    # Standard LLM
    print("\n--- Standard LLM ---")
    config = create_standard_config(num_layers=3)
    print(f"Config: {config.describe()}")

    set_seed(exp_config.seed)
    model = DeepSupervisionTransformer(
        vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
        exit_layer=1, routing_threshold=0.8
    )
    trainer = Trainer(config, vocab_size=vocab_size, device=exp_config.device)

    result = run_experiment(
        name="Standard LLM", config=config, model=model, trainer=trainer,
        train_batches=train_batches, val_batches=val_batches,
        lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
    )
    results.append(result)

    # Deep Supervision
    print("\n--- Deep Supervision ---")
    config = create_deep_supervision_config(num_layers=3)
    print(f"Config: {config.describe()}")

    set_seed(exp_config.seed)
    model = DeepSupervisionTransformer(
        vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
        exit_layer=1, routing_threshold=0.8
    )
    trainer = Trainer(config, vocab_size=vocab_size, device=exp_config.device)

    result = run_experiment(
        name="Deep Supervision", config=config, model=model, trainer=trainer,
        train_batches=train_batches, val_batches=val_batches,
        lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
    )
    results.append(result)

    # =========================================================================
    # Part 2: With Early Exit
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: With Early Exit")
    print("=" * 70)

    for alpha, name in [(0.5, "α=0.5"), (0.7, "α=0.7"), (0.8, "α=0.8")]:
        print(f"\n--- Asymmetric ({name}) + Early Exit ---")
        config = TrainingConfig(
            layer_weights={1: alpha, 2: 0, 3: 1-alpha},
            routing_threshold=0.95
        )
        print(f"Config: {config.describe()}")

        set_seed(exp_config.seed)
        model = DeepSupervisionTransformer(
            vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
            exit_layer=1, routing_threshold=0.8
        )
        trainer = Trainer(config, vocab_size=vocab_size, device=exp_config.device)

        result = run_experiment(
            name=f"Asymmetric ({name})", config=config, model=model, trainer=trainer,
            train_batches=train_batches, val_batches=val_batches,
            lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
        )
        results.append(result)

    # =========================================================================
    # Part 3: With Layer-wise Learning Rate
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: With Layer-wise Learning Rate (Discriminative Fine-Tuning)")
    print("=" * 70)

    for lr_name, lr_scales in [
        ("Decreasing LR", {1: 1.0, 2: 0.5, 3: 0.1}),
        ("Increasing LR", {1: 0.1, 2: 0.5, 3: 1.0}),
    ]:
        print(f"\n--- {lr_name} ---")
        config = TrainingConfig(
            layer_weights={1: 0.7, 2: 0, 3: 0.3},
            layer_lr_scales=lr_scales,
            routing_threshold=0.95
        )
        print(f"Config: {config.describe()}")

        set_seed(exp_config.seed)
        model = DeepSupervisionTransformer(
            vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
            exit_layer=1, routing_threshold=0.8
        )
        trainer = Trainer(config, vocab_size=vocab_size, device=exp_config.device)

        result = run_experiment(
            name=f"Layer-wise LR: {lr_name}", config=config, model=model, trainer=trainer,
            train_batches=train_batches, val_batches=val_batches,
            lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
        )
        results.append(result)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results.sort(key=lambda x: x['best_ppl'])

    print(f"\n{'Rank':<5} {'Model':<35} {'PPL':>8} {'Shallow%':>10} {'Compute%':>10}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        shallow = r.get('shallow_ratio', 0) * 100
        compute = r.get('compute_cost', 1) * 100
        print(f"{i:<5} {r['name']:<35} {r['best_ppl']:>8.2f} {shallow:>9.1f}% {compute:>9.1f}%")

    # Best model
    best = results[0]
    print(f"\n🥇 Best: {best['name']} (PPL={best['best_ppl']:.2f})")

    # Key insights
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")

    std = next((r for r in results if r['name'] == 'Standard LLM'), None)
    if std:
        improvement = (std['best_ppl'] - best['best_ppl']) / std['best_ppl'] * 100
        compute_save = (1 - best.get('compute_cost', 1)) * 100
        print(f"  - Best model: {improvement:.1f}% better than Standard LLM")
        if compute_save > 0:
            print(f"  - Compute savings: {compute_save:.1f}%")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
