#!/usr/bin/env python3
"""
Threshold Sweep Experiment

Train DeepSupervision once, then evaluate with different routing thresholds.
This tests inference-time early exit with varying confidence thresholds.
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
    DeepSupervisionTransformer,
    UniversalConfig,
    UniversalTrainer,
)
from experiments import set_seed, prepare_wikitext_data


def evaluate_with_threshold(
    model: DeepSupervisionTransformer,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
) -> Dict:
    """Evaluate model with a specific routing threshold."""
    # Temporarily change the threshold
    original_threshold = model.routing_threshold
    model.routing_threshold = threshold
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_tokens = 0
    total_shallow_ratio = 0.0
    total_compute_cost = 0.0
    
    with torch.no_grad():
        for x, y in val_batches:
            output, stats = model.forward_inference(x)
            
            loss = F.cross_entropy(
                output.view(-1, model.vocab_size),
                y.view(-1)
            )
            val_loss += loss.item()
            
            preds = output.argmax(dim=-1)
            val_correct += (preds == y).sum().item()
            val_tokens += y.numel()
            total_shallow_ratio += stats['shallow_ratio']
            total_compute_cost += stats['compute_cost']
    
    # Restore original threshold
    model.routing_threshold = original_threshold
    
    val_ppl = np.exp(val_loss / len(val_batches))
    val_acc = val_correct / val_tokens
    avg_shallow = total_shallow_ratio / len(val_batches)
    avg_compute = total_compute_cost / len(val_batches)
    
    return {
        'threshold': threshold,
        'ppl': val_ppl,
        'acc': val_acc,
        'shallow_ratio': avg_shallow,
        'compute_cost': avg_compute,
    }


def main() -> None:
    print("=" * 70)
    print("THRESHOLD SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    TRAIN_CHARS = 500000
    VAL_CHARS = 50000
    SEQ_LEN = 128
    BATCH_SIZE = 32
    DIM = 32
    NUM_HEADS = 2
    NUM_LAYERS = 4
    LR = 5e-4
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
    
    # =========================================================================
    # Phase 1: Train DeepSupervision (with routing disabled for training)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Training DeepSupervision (routing_threshold=0, full layers)")
    print("=" * 70)
    
    set_seed(SEED)
    model = DeepSupervisionTransformer(
        vocab_size, DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        exit_layer=2, routing_threshold=0  # No early exit during training
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Exit layer: 2 (of {NUM_LAYERS})")
    
    # Config: α=0.7 on exit layer (L2), (1-α)=0.3 on final layer (L4)
    config = UniversalConfig(
        layer_weights={1: 0, 2: 0.7, 3: 0, 4: 0.3},
        routing_threshold=0,  # No early exit during training
        exit_layer=2
    )
    trainer = UniversalTrainer(config, vocab_size=vocab_size)
    optimizer = trainer.create_optimizer(model, base_lr=LR)
    
    best_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        train_loss, _ = trainer.train_epoch(
            model, train_batches, optimizer, 1.0, epoch, MAX_EPOCHS
        )
        
        # Evaluate with no routing (full layers)
        stats = evaluate_with_threshold(model, val_batches, threshold=1.0)
        epoch_time = time.time() - epoch_start
        
        print(f"  Epoch {epoch+1}: Train PPL={np.exp(train_loss):.2f}, "
              f"Val PPL={stats['ppl']:.2f} ({epoch_time:.1f}s)")
        
        if stats['ppl'] < best_ppl:
            best_ppl = stats['ppl']
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model state
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print("  >>> Early stopping")
                break
    
    # Restore best model
    model.load_state_dict(best_state)
    print(f"\nTraining complete. Best PPL: {best_ppl:.2f} (epoch {best_epoch})")
    
    # =========================================================================
    # Phase 2: Threshold Sweep
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Threshold Sweep (inference-time early exit)")
    print("=" * 70)
    
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    
    print(f"\n{'Threshold':<12} {'PPL':>8} {'Shallow%':>10} {'Compute%':>10} {'PPL Δ':>10}")
    print("-" * 52)
    
    baseline_ppl = None
    
    for threshold in thresholds:
        stats = evaluate_with_threshold(model, val_batches, threshold)
        results.append(stats)
        
        if threshold == 1.0:
            baseline_ppl = stats['ppl']
        
        ppl_delta = ""
        if baseline_ppl and threshold < 1.0:
            delta = (stats['ppl'] - baseline_ppl) / baseline_ppl * 100
            ppl_delta = f"{delta:+.1f}%"
        
        print(f"{threshold:<12.1f} {stats['ppl']:>8.2f} "
              f"{stats['shallow_ratio']*100:>9.1f}% "
              f"{stats['compute_cost']*100:>9.1f}% "
              f"{ppl_delta:>10}")
    
    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Find best trade-off points
    baseline = results[-1]  # threshold=1.0
    
    print("\nKey findings:")
    print(f"  - Baseline (no early exit): PPL={baseline['ppl']:.2f}, Compute=100%")
    
    # Find threshold with best efficiency (lowest compute while PPL < baseline + 5%)
    ppl_tolerance = baseline['ppl'] * 1.05  # 5% degradation allowed
    efficient_results = [r for r in results if r['ppl'] <= ppl_tolerance]
    if efficient_results:
        best_efficient = min(efficient_results, key=lambda r: r['compute_cost'])
        print(f"  - Best efficiency (PPL ≤ {ppl_tolerance:.2f}): "
              f"threshold={best_efficient['threshold']:.1f}, "
              f"PPL={best_efficient['ppl']:.2f}, "
              f"Compute={best_efficient['compute_cost']*100:.1f}%")
        
        compute_save = (1 - best_efficient['compute_cost']) * 100
        ppl_increase = (best_efficient['ppl'] - baseline['ppl']) / baseline['ppl'] * 100
        print(f"  - Trade-off: {compute_save:.1f}% compute saved for {ppl_increase:.1f}% PPL increase")
    
    # Find threshold where PPL starts degrading significantly (>10%)
    for r in results:
        if r['ppl'] > baseline['ppl'] * 1.10:
            print(f"  - PPL degrades >10% at threshold={r['threshold']:.1f}")
            break
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
