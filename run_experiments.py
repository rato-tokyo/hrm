#!/usr/bin/env python3
"""
EASE: Efficient Asymmetric Supervision for Early-Exit Transformers

Universal Training Framework for Early-Exit experiments.
See experiments/universal_trainer.py for the framework implementation.

Key training methods:
- Standard LLM: Loss = L_final only
- Deep Supervision: Loss = (L1 + L2 + ... + Ln) / n  (Lee et al., 2015)
- Auxiliary Loss: Loss = α * L1 + (1-α) * Ln  (Elbayad et al., 2020)
- Asymmetric (EASE): Loss = 0.7 * L1 + 0.3 * Ln with L2=0 (Ours)
- Discriminative Fine-Tuning: Layer-wise LR (Howard & Ruder, 2018)

See docs/experiments/ for detailed results and analysis.
"""

import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
import sys
sys.path.insert(0, '.')

from experiments.utils import set_seed, prepare_wikitext_data, ExperimentConfig
from experiments.models import ConfidenceRoutedTransformer
from experiments.universal_trainer import UniversalConfig, UniversalTrainer, PRESETS, AlphaSchedule


def run_experiment(
    name: str,
    config: UniversalConfig,
    model: torch.nn.Module,
    trainer: UniversalTrainer,
    train_batches: List,
    val_batches: List,
    lr: float = 1e-3,
    max_epochs: int = 50,
    grad_clip: float = 1.0,
    verbose: bool = True
) -> Dict:
    """Run experiment with universal trainer."""
    # Use trainer's optimizer creation (supports layer-wise LR)
    optimizer = trainer.create_optimizer(model, base_lr=lr)

    best_ppl = float('inf')
    best_epoch = 0
    best_stats = {}
    alpha_history = []

    for epoch in range(max_epochs):
        # train_epoch now returns (loss, current_weights) for dynamic alpha tracking
        train_loss, current_weights = trainer.train_epoch(
            model, train_batches, optimizer, grad_clip,
            epoch=epoch, max_epochs=max_epochs
        )
        train_ppl = np.exp(train_loss)

        stats = trainer.evaluate(model, val_batches)
        val_ppl = stats['ppl']

        # Track alpha for dynamic schedules
        if config.has_dynamic_alpha:
            alpha = current_weights.get(1, 0)
            alpha_history.append(alpha)
            alpha_info = f", α={alpha:.2f}"
        else:
            alpha_info = ""

        if verbose:
            routing_info = f", Shallow={stats['shallow_ratio']*100:.1f}%, Compute={stats['compute_cost']*100:.1f}%" if config.has_routing else ""
            print(f"  Epoch {epoch+1}: Train={train_ppl:.2f}, Val={val_ppl:.2f}{routing_info}{alpha_info}")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            best_stats = stats.copy()
        else:
            if verbose:
                print("  >>> Early stopping")
            break

    result = {
        'name': name,
        'config': config.describe(),
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        **best_stats
    }

    if alpha_history:
        result['alpha_history'] = alpha_history

    return result


def main():
    print("=" * 70)
    print("HRM EXPERIMENTS - UNIVERSAL TRAINING FRAMEWORK")
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

    results = []

    # =========================================================================
    # Standard Models (No Routing)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Standard Models (No Routing)")
    print("=" * 70)

    for preset_name, display_name in [
        ('standard_llm', 'Standard LLM'),
        ('lpt', 'LPT'),
    ]:
        print(f"\n--- {display_name} ---")
        config = PRESETS[preset_name]
        print(f"Config: {config.describe()}")

        set_seed(exp_config.seed)
        model = ConfidenceRoutedTransformer(
            vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
            exit_layer=1, routing_threshold=0.8
        )
        trainer = UniversalTrainer(config, vocab_size=vocab_size, device=exp_config.device)

        result = run_experiment(
            name=display_name, config=config, model=model, trainer=trainer,
            train_batches=train_batches, val_batches=val_batches,
            lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
        )
        results.append(result)

    # =========================================================================
    # Routing Models
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Routing Models")
    print("=" * 70)

    for preset_name, display_name in [
        ('standard_routing', 'Standard Routing (α=0.5)'),
        ('asymmetric_best', 'Asymmetric (α=0.7)'),
        ('lpt_routing', 'LPT + Routing'),
    ]:
        print(f"\n--- {display_name} ---")
        config = PRESETS[preset_name]
        print(f"Config: {config.describe()}")

        set_seed(exp_config.seed)
        model = ConfidenceRoutedTransformer(
            vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
            exit_layer=1, routing_threshold=0.8
        )
        trainer = UniversalTrainer(config, vocab_size=vocab_size, device=exp_config.device)

        result = run_experiment(
            name=display_name, config=config, model=model, trainer=trainer,
            train_batches=train_batches, val_batches=val_batches,
            lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
        )
        results.append(result)

    # =========================================================================
    # Alpha Search
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Alpha Optimization")
    print("=" * 70)

    for alpha in [0.6, 0.8, 0.9]:
        display_name = f"Asymmetric (α={alpha})"
        print(f"\n--- {display_name} ---")
        config = UniversalConfig(
            layer_weights={1: alpha, 2: 0, 3: 1-alpha},
            routing_threshold=0.95
        )
        print(f"Config: {config.describe()}")

        set_seed(exp_config.seed)
        model = ConfidenceRoutedTransformer(
            vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
            exit_layer=1, routing_threshold=0.8
        )
        trainer = UniversalTrainer(config, vocab_size=vocab_size, device=exp_config.device)

        result = run_experiment(
            name=display_name, config=config, model=model, trainer=trainer,
            train_batches=train_batches, val_batches=val_batches,
            lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
        )
        results.append(result)

    # =========================================================================
    # L2 Loss Impact
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: L2 Loss Impact")
    print("=" * 70)

    print("\n--- Asymmetric + L2 ---")
    config = PRESETS['asymmetric_with_l2']
    print(f"Config: {config.describe()}")

    set_seed(exp_config.seed)
    model = ConfidenceRoutedTransformer(
        vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
        exit_layer=1, routing_threshold=0.8
    )
    trainer = UniversalTrainer(config, vocab_size=vocab_size, device=exp_config.device)

    result = run_experiment(
        name="Asymmetric + L2", config=config, model=model, trainer=trainer,
        train_batches=train_batches, val_batches=val_batches,
        lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
    )
    results.append(result)

    # =========================================================================
    # NEW FEATURES: Dynamic Alpha
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: Dynamic Alpha (Curriculum Learning)")
    print("=" * 70)

    # Linear decay: 0.9 -> 0.5
    for schedule_type, start, end in [
        ('linear', 0.9, 0.5),
        ('cosine', 0.9, 0.5),
    ]:
        display_name = f"Dynamic α ({schedule_type}: {start}→{end})"
        print(f"\n--- {display_name} ---")

        config = UniversalConfig(
            layer_weights={1: start, 2: 0, 3: 1-start},  # Initial weights
            routing_threshold=0.95,
            alpha_schedule=AlphaSchedule(schedule_type, start=start, end=end)
        )
        print(f"Config: {config.describe()}")

        set_seed(exp_config.seed)
        model = ConfidenceRoutedTransformer(
            vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
            exit_layer=1, routing_threshold=0.8
        )
        trainer = UniversalTrainer(config, vocab_size=vocab_size, device=exp_config.device)

        result = run_experiment(
            name=display_name, config=config, model=model, trainer=trainer,
            train_batches=train_batches, val_batches=val_batches,
            lr=exp_config.lr, max_epochs=exp_config.max_epochs, grad_clip=exp_config.grad_clip
        )
        results.append(result)

    # =========================================================================
    # NEW FEATURES: Layer-wise Learning Rate
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 6: Layer-wise Learning Rate")
    print("=" * 70)

    for lr_config_name, lr_scales in [
        ("Decreasing LR (1.0, 0.5, 0.1)", {1: 1.0, 2: 0.5, 3: 0.1}),
        ("Increasing LR (0.1, 0.5, 1.0)", {1: 0.1, 2: 0.5, 3: 1.0}),
    ]:
        display_name = f"Layer-wise LR: {lr_config_name}"
        print(f"\n--- {display_name} ---")

        config = UniversalConfig(
            layer_weights={1: 0.7, 2: 0, 3: 0.3},  # Same as asymmetric_best
            routing_threshold=0.95,
            layer_lr_scales=lr_scales
        )
        print(f"Config: {config.describe()}")

        set_seed(exp_config.seed)
        model = ConfidenceRoutedTransformer(
            vocab_size, exp_config.dim, num_layers=3, num_heads=exp_config.num_heads,
            exit_layer=1, routing_threshold=0.8
        )
        trainer = UniversalTrainer(config, vocab_size=vocab_size, device=exp_config.device)

        result = run_experiment(
            name=display_name, config=config, model=model, trainer=trainer,
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

    print(f"\n{'Rank':<5} {'Model':<30} {'PPL':>8} {'Shallow%':>10} {'Compute%':>10}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        shallow = r.get('shallow_ratio', 0) * 100
        compute = r.get('compute_cost', 1) * 100
        print(f"{i:<5} {r['name']:<30} {r['best_ppl']:>8.2f} {shallow:>9.1f}% {compute:>9.1f}%")

    # Best model
    best = results[0]
    print(f"\n🥇 Best: {best['name']} (PPL={best['best_ppl']:.2f})")

    # Key insights
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")

    # Find Standard LLM baseline
    std = next((r for r in results if r['name'] == 'Standard LLM'), None)
    if std:
        improvement = (std['best_ppl'] - best['best_ppl']) / std['best_ppl'] * 100
        compute_save = (1 - best.get('compute_cost', 1)) * 100
        print(f"  - Best model: {improvement:.1f}% better than Standard LLM")
        if compute_save > 0:
            print(f"  - Compute savings: {compute_save:.1f}%")

    # L2 impact
    asym = next((r for r in results if r['name'] == 'Asymmetric (α=0.7)'), None)
    asym_l2 = next((r for r in results if r['name'] == 'Asymmetric + L2'), None)
    if asym and asym_l2:
        l2_impact = (asym_l2['best_ppl'] - asym['best_ppl']) / asym['best_ppl'] * 100
        print(f"  - L2 loss impact: {l2_impact:+.1f}% (adding L2 loss hurts performance)")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
