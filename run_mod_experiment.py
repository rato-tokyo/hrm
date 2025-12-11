#!/usr/bin/env python3
"""
Mixture-of-Depths Experiment

Compare MoD transformer with baseline and EASE models.

Usage:
    python run_mod_experiment.py
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple

from ease import (
    StandardTransformer,
    DeepSupervisionTransformer,
    MoDTransformer,
    UniversalConfig,
    UniversalTrainer,
)
from experiments import set_seed, prepare_wikitext_data


def train_mod_model(
    model: MoDTransformer,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    lr: float = 1e-3,
    max_epochs: int = 50,
    grad_clip: float = 1.0,
    aux_loss_weight: float = 0.01,
    verbose: bool = True,
) -> Dict:
    """Train MoD model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_ppl = float('inf')
    best_epoch = 0
    best_stats: Dict = {}
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        total_routing_ratio = 0.0
        epoch_start = time.time()
        num_batches = len(train_batches)

        for batch_idx, (x, y) in enumerate(train_batches):
            optimizer.zero_grad()

            outputs = model.forward_train(x)
            logits = outputs['logits']
            aux_loss = outputs['aux_loss']

            # Language modeling loss
            lm_loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                y.view(-1)
            )

            # Combined loss
            loss = lm_loss + aux_loss_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += lm_loss.item()
            total_aux_loss += aux_loss.item()
            total_routing_ratio += outputs['routing_ratio']

            # Progress log every 50 batches
            if verbose and (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                print(f"    Batch {batch_idx+1}/{num_batches} "
                      f"(ETA: {eta:.0f}s, Loss: {lm_loss.item():.3f})")

        train_ppl = np.exp(total_loss / len(train_batches))
        avg_aux_loss = total_aux_loss / len(train_batches)
        epoch_time = time.time() - epoch_start

        # Evaluate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0
        val_routing = 0.0

        with torch.no_grad():
            for x, y in val_batches:
                outputs = model.forward_train(x)
                logits = outputs['logits']

                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    y.view(-1)
                )
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_tokens += y.numel()
                val_routing += outputs['routing_ratio']

        val_ppl = np.exp(val_loss / len(val_batches))
        val_acc = val_correct / val_tokens
        avg_val_routing = val_routing / len(val_batches)

        if verbose:
            print(f"  Epoch {epoch+1}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f}, "
                  f"Routing={avg_val_routing*100:.1f}%, Aux={avg_aux_loss:.4f} ({epoch_time:.1f}s)")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            best_stats = {
                'ppl': val_ppl,
                'acc': val_acc,
                'routing_ratio': avg_val_routing,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                if verbose:
                    print("  >>> Early stopping")
                break

    return {
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        **best_stats,
    }


def train_baseline(
    model: StandardTransformer,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    lr: float = 1e-3,
    max_epochs: int = 50,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> Dict:
    """Train baseline (standard) model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_ppl = float('inf')
    best_epoch = 0
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        num_batches = len(train_batches)

        for batch_idx, (x, y) in enumerate(train_batches):
            optimizer.zero_grad()

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                y.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

            # Progress log every 50 batches
            if verbose and (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                print(f"    Batch {batch_idx+1}/{num_batches} "
                      f"(ETA: {eta:.0f}s, Loss: {loss.item():.3f})")

        train_ppl = np.exp(total_loss / len(train_batches))
        epoch_time = time.time() - epoch_start

        # Evaluate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0

        with torch.no_grad():
            for x, y in val_batches:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    y.view(-1)
                )
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_tokens += y.numel()

        val_ppl = np.exp(val_loss / len(val_batches))
        val_acc = val_correct / val_tokens

        if verbose:
            print(f"  Epoch {epoch+1}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f} ({epoch_time:.1f}s)")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                if verbose:
                    print("  >>> Early stopping")
                break

    return {
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        'acc': best_acc,
    }


def main() -> None:
    print("=" * 70)
    print("MIXTURE-OF-DEPTHS EXPERIMENT")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration - balanced for fair comparison
    TRAIN_CHARS = 500000
    VAL_CHARS = 50000
    SEQ_LEN = 128
    BATCH_SIZE = 32
    DIM = 32           # Smaller model to reduce overfitting
    NUM_HEADS = 2
    NUM_LAYERS = 4     # Fewer layers
    LR = 5e-4          # Lower LR
    MAX_EPOCHS = 30
    SEED = 42

    print(f"\nConfig: {TRAIN_CHARS:,} train chars, seq_len={SEQ_LEN}, "
          f"dim={DIM}, layers={NUM_LAYERS}")

    # Prepare data
    train_batches, val_batches, vocab_size, _ = prepare_wikitext_data(
        TRAIN_CHARS, VAL_CHARS, SEQ_LEN, BATCH_SIZE
    )
    print(f"Data: {len(train_batches)} train batches, {len(val_batches)} val batches")
    print(f"Vocab size: {vocab_size}")

    results: List[Dict] = []

    # =========================================================================
    # Baseline: Standard Transformer (6 layers)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. Baseline: Standard Transformer (6 layers)")
    print("=" * 70)

    set_seed(SEED)
    baseline_model = StandardTransformer(
        vocab_size, DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS
    )
    print(f"Parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")

    result = train_baseline(
        baseline_model, train_batches, val_batches,
        lr=LR, max_epochs=MAX_EPOCHS
    )
    result['name'] = 'Baseline (6L)'
    result['compute_cost'] = 1.0
    results.append(result)

    # =========================================================================
    # MoD: 50% capacity
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. MoD: 50% capacity, every other layer")
    print("=" * 70)

    set_seed(SEED)
    mod_model_50 = MoDTransformer(
        vocab_size, DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        capacity=0.5, route_every_n=2
    )
    print(f"Capacity: 50% ({int(SEQ_LEN * 0.5)} tokens per routing layer)")

    result = train_mod_model(
        mod_model_50, train_batches, val_batches,
        lr=LR, max_epochs=MAX_EPOCHS, aux_loss_weight=0.01
    )
    result['name'] = 'MoD (50%)'
    result['compute_cost'] = (3 + 3 * 0.5) / 6
    results.append(result)

    # =========================================================================
    # DeepSupervision: α-weighted loss distribution (for comparison)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. DeepSupervision (α=0.7) - 4 layers")
    print("=" * 70)

    set_seed(SEED)
    ds_model = DeepSupervisionTransformer(
        vocab_size, DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        exit_layer=1, routing_threshold=0.95
    )
    # DeepSupervision config: α=0.7 on exit layer (L1), (1-α)=0.3 on final layer (L4)
    # All tokens contribute to both losses
    config = UniversalConfig(
        layer_weights={1: 0.7, 2: 0, 3: 0, 4: 0.3},
        routing_threshold=0.95,
        exit_layer=1
    )
    trainer = UniversalTrainer(config, vocab_size=vocab_size)

    optimizer = trainer.create_optimizer(ds_model, base_lr=LR)

    best_ppl = float('inf')
    best_stats: Dict = {}
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        train_loss, _ = trainer.train_epoch(
            ds_model, train_batches, optimizer, 1.0, epoch, MAX_EPOCHS
        )
        stats = trainer.evaluate(ds_model, val_batches)
        epoch_time = time.time() - epoch_start

        print(f"  Epoch {epoch+1}: Train PPL={np.exp(train_loss):.2f}, "
              f"Val PPL={stats['ppl']:.2f}, Shallow={stats['shallow_ratio']*100:.1f}% ({epoch_time:.1f}s)")

        if stats['ppl'] < best_ppl:
            best_ppl = stats['ppl']
            best_stats = stats.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print("  >>> Early stopping")
                break

    results.append({
        'name': 'DeepSupervision (α=0.7)',
        'best_ppl': best_ppl,
        **best_stats,
    })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results.sort(key=lambda x: x['best_ppl'])

    print(f"\n{'Rank':<5} {'Model':<20} {'PPL':>8} {'Compute':>10} {'vs Base':>10}")
    print("-" * 60)

    baseline_ppl = next(r['best_ppl'] for r in results if 'Baseline' in r['name'])

    for i, r in enumerate(results, 1):
        compute = r.get('compute_cost', 1.0) * 100
        improvement = (baseline_ppl - r['best_ppl']) / baseline_ppl * 100
        print(f"{i:<5} {r['name']:<20} {r['best_ppl']:>8.2f} {compute:>9.1f}% {improvement:>+9.1f}%")

    print("\n" + "-" * 60)
    print("KEY FINDINGS:")

    best = results[0]
    print(f"  - Best model: {best['name']} (PPL={best['best_ppl']:.2f})")

    mod_12 = next((r for r in results if '12.5%' in r['name']), None)
    if mod_12:
        improvement = (baseline_ppl - mod_12['best_ppl']) / baseline_ppl * 100
        compute_save = (1 - mod_12.get('compute_cost', 1.0)) * 100
        print(f"  - MoD 12.5%: {improvement:+.1f}% PPL, {compute_save:.1f}% compute savings")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
