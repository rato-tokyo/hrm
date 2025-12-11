#!/usr/bin/env python3
"""
Token-Level Routing Experiment

Compare TokenRoutedTransformer (mask-based token routing) with previous models.

Key difference from DeepSupervisionTransformer:
- DeepSupervision: All tokens contribute to both shallow/deep loss (α-weighted)
- TokenRouted: Each token contributes to ONLY its routed path's loss (no α needed)

Usage:
    python run_token_routing_experiment.py
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple

from ease import TokenRoutedTransformer
from experiments import set_seed, prepare_wikitext_data


def train_token_routed_model(
    model: TokenRoutedTransformer,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    lr: float = 1e-3,
    max_epochs: int = 50,
    grad_clip: float = 1.0,
    alpha: float = 0.7,  # Weight for shallow loss when computing combined loss
    verbose: bool = True,
) -> Dict:
    """
    Train TokenRoutedTransformer with token-level loss routing.

    Loss computation:
    - Tokens with high confidence (exit early): shallow_loss only
    - Tokens with low confidence (go deep): deep_loss only
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_ppl = float('inf')
    best_epoch = 0
    best_stats: Dict = {}
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_shallow_ratio = 0.0
        epoch_start = time.time()
        num_batches = len(train_batches)

        for batch_idx, (x, y) in enumerate(train_batches):
            optimizer.zero_grad()

            outputs = model.forward_train(x)
            shallow_logits = outputs['shallow_logits']  # [batch, seq_len, vocab]
            deep_logits = outputs['deep_logits']        # [batch, seq_len, vocab]
            exit_mask = outputs['exit_mask']            # [batch, seq_len]

            # Token-level loss: each token uses only its routed path
            # Flatten for cross_entropy
            batch_size, seq_len, vocab_size = shallow_logits.shape
            y_flat = y.view(-1)
            exit_mask_flat = exit_mask.view(-1)

            shallow_logits_flat = shallow_logits.view(-1, vocab_size)
            deep_logits_flat = deep_logits.view(-1, vocab_size)

            # Compute per-token loss
            shallow_loss_per_token = F.cross_entropy(
                shallow_logits_flat, y_flat, reduction='none'
            )
            deep_loss_per_token = F.cross_entropy(
                deep_logits_flat, y_flat, reduction='none'
            )

            # Select loss based on routing
            # exit_mask=True -> use shallow_loss, exit_mask=False -> use deep_loss
            loss_per_token = torch.where(
                exit_mask_flat,
                shallow_loss_per_token,
                deep_loss_per_token
            )

            # Average loss
            loss = loss_per_token.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_shallow_ratio += outputs['shallow_ratio']

            # Progress log every 50 batches
            if verbose and (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                print(f"    Batch {batch_idx+1}/{num_batches} "
                      f"(ETA: {eta:.0f}s, Loss: {loss.item():.3f})")

        train_ppl = np.exp(total_loss / len(train_batches))
        avg_shallow_ratio = total_shallow_ratio / len(train_batches)
        epoch_time = time.time() - epoch_start

        # Evaluate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0
        val_shallow_ratio = 0.0
        val_compute_cost = 0.0

        with torch.no_grad():
            for x, y in val_batches:
                # Use inference forward with hard routing
                output, stats = model.forward_inference(x)

                loss = F.cross_entropy(
                    output.view(-1, model.vocab_size),
                    y.view(-1)
                )
                val_loss += loss.item()

                preds = output.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_tokens += y.numel()
                val_shallow_ratio += stats['shallow_ratio']
                val_compute_cost += stats['compute_cost']

        val_ppl = np.exp(val_loss / len(val_batches))
        val_acc = val_correct / val_tokens
        avg_val_shallow = val_shallow_ratio / len(val_batches)
        avg_val_compute = val_compute_cost / len(val_batches)

        if verbose:
            print(f"  Epoch {epoch+1}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f}, "
                  f"Shallow={avg_val_shallow*100:.1f}%, Compute={avg_val_compute*100:.1f}% ({epoch_time:.1f}s)")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            best_stats = {
                'ppl': val_ppl,
                'acc': val_acc,
                'shallow_ratio': avg_val_shallow,
                'compute_cost': avg_val_compute,
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


def train_token_routed_with_curriculum(
    model: TokenRoutedTransformer,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    lr: float = 1e-3,
    max_epochs: int = 50,
    grad_clip: float = 1.0,
    warmup_epochs: int = 5,  # Epochs of joint training before routing
    shallow_weight: float = 0.7,  # Alpha for shallow loss during warmup
    verbose: bool = True,
) -> Dict:
    """
    Train TokenRoutedTransformer with curriculum learning.

    Phase 1 (warmup_epochs): Train both paths jointly (like EASE)
    Phase 2 (remaining): Use token-level routing

    This addresses the cold-start problem where shallow layer isn't trained
    when all tokens initially route to deep path.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_ppl = float('inf')
    best_epoch = 0
    best_stats: Dict = {}
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_shallow_ratio = 0.0
        epoch_start = time.time()
        num_batches = len(train_batches)

        # Determine training mode
        use_curriculum = (epoch < warmup_epochs)

        for batch_idx, (x, y) in enumerate(train_batches):
            optimizer.zero_grad()

            outputs = model.forward_train(x)
            shallow_logits = outputs['shallow_logits']
            deep_logits = outputs['deep_logits']
            exit_mask = outputs['exit_mask']

            batch_size, seq_len, vocab_size = shallow_logits.shape
            y_flat = y.view(-1)

            shallow_logits_flat = shallow_logits.view(-1, vocab_size)
            deep_logits_flat = deep_logits.view(-1, vocab_size)

            if use_curriculum:
                # Phase 1: Joint training (like EASE/Deep Supervision)
                shallow_loss = F.cross_entropy(shallow_logits_flat, y_flat)
                deep_loss = F.cross_entropy(deep_logits_flat, y_flat)
                loss = shallow_weight * shallow_loss + (1 - shallow_weight) * deep_loss
            else:
                # Phase 2: Token-level routing
                exit_mask_flat = exit_mask.view(-1)

                shallow_loss_per_token = F.cross_entropy(
                    shallow_logits_flat, y_flat, reduction='none'
                )
                deep_loss_per_token = F.cross_entropy(
                    deep_logits_flat, y_flat, reduction='none'
                )

                loss_per_token = torch.where(
                    exit_mask_flat,
                    shallow_loss_per_token,
                    deep_loss_per_token
                )
                loss = loss_per_token.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_shallow_ratio += outputs['shallow_ratio']

            if verbose and (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                mode = "warmup" if use_curriculum else "routing"
                print(f"    Batch {batch_idx+1}/{num_batches} "
                      f"(ETA: {eta:.0f}s, Loss: {loss.item():.3f}, {mode})")

        train_ppl = np.exp(total_loss / len(train_batches))
        epoch_time = time.time() - epoch_start

        # Evaluate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0
        val_shallow_ratio = 0.0
        val_compute_cost = 0.0

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
                val_shallow_ratio += stats['shallow_ratio']
                val_compute_cost += stats['compute_cost']

        val_ppl = np.exp(val_loss / len(val_batches))
        val_acc = val_correct / val_tokens
        avg_val_shallow = val_shallow_ratio / len(val_batches)
        avg_val_compute = val_compute_cost / len(val_batches)

        mode_str = "[warmup]" if use_curriculum else "[routing]"
        if verbose:
            print(f"  Epoch {epoch+1} {mode_str}: Train PPL={train_ppl:.2f}, Val PPL={val_ppl:.2f}, "
                  f"Shallow={avg_val_shallow*100:.1f}%, Compute={avg_val_compute*100:.1f}% ({epoch_time:.1f}s)")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            best_stats = {
                'ppl': val_ppl,
                'acc': val_acc,
                'shallow_ratio': avg_val_shallow,
                'compute_cost': avg_val_compute,
            }
            patience_counter = 0
        else:
            # Only start patience after warmup
            if not use_curriculum:
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


def main() -> None:
    print("=" * 70)
    print("TOKEN-LEVEL ROUTING EXPERIMENT")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration - same as previous experiment
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

    results: List[Dict] = []

    # =========================================================================
    # TokenRouted with Curriculum Learning (warmup + routing)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TokenRoutedTransformer with Curriculum Learning")
    print("  Phase 1 (5 epochs): Joint training like EASE")
    print("  Phase 2: Token-level routing")
    print("=" * 70)

    set_seed(SEED)
    model = TokenRoutedTransformer(
        vocab_size, DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        exit_layer=2, routing_threshold=0.7  # exit at layer 2 (halfway through 4 layers)
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    result = train_token_routed_with_curriculum(
        model, train_batches, val_batches,
        lr=LR, max_epochs=MAX_EPOCHS,
        warmup_epochs=5,
        shallow_weight=0.7
    )
    result['name'] = 'TokenRouted+Curriculum'
    results.append(result)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  {'Model':<30} {'PPL':>8} {'Shallow':>10} {'Compute':>10}")
    print("-" * 70)

    for r in results:
        print(f"  {r['name']:<30} {r['best_ppl']:>8.2f} "
              f"{r.get('shallow_ratio', 0)*100:>9.1f}% "
              f"{r.get('compute_cost', 1.0)*100:>9.1f}%")

    print("\n" + "-" * 70)
    print("COMPARISON WITH PREVIOUS RESULTS:")
    print("-" * 70)
    print(f"  {'Model':<30} {'PPL':>8} {'Shallow':>10} {'Compute':>10}")
    print("-" * 70)
    print(f"  {'DeepSupervision (α=0.7)':<30} {'12.41':>8} {'0.1%':>10} {'99.9%':>10}")
    print(f"  {'MoD (50%)':<30} {'12.64':>8} {'50.0%':>10} {'75.0%':>10}")
    print(f"  {'Baseline (4L)':<30} {'16.31':>8} {'-':>10} {'100.0%':>10}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
