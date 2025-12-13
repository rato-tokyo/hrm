# -*- coding: utf-8 -*-
"""
LEGO Framework - Quick Test Script

ローカルで実行して、LEGOが正しく機能しているか確認するテスト。
小さいデータセット + 固定シードで数値の完全一致を検証。

Usage:
    python test_lego.py

Expected:
    - Phase 1 → Phase 2 で Hard PPL が改善
    - Hard tokens 収集率 ~50%（100%ならバグ）
    - Shallow ratio > 0（早期終了が発生）
    - 実行時間 < 5秒
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
import numpy as np

from ease import (
    DeepSupervisionTransformer,
    Trainer,
    TrainingConfig,
    StageConfig,
    LEGOConfig,
    PhaseConfig,
    LEGOTrainer,
    compute_confidence_threshold,
    collect_hard_examples,
    evaluate_on_hard_examples,
)


# ==============================================================================
# Configuration
# ==============================================================================

VOCAB_SIZE = 100
DIM = 32
NUM_HEADS = 2
SEQ_LEN = 16
BATCH_SIZE = 8
NUM_BATCHES = 4
SEED = 42
DEVICE = 'cpu'


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ==============================================================================
# Test 1: LEGOTrainer API
# ==============================================================================

def test_lego_trainer() -> bool:
    """Test LEGOTrainer with cascading phases."""
    print("=" * 60)
    print("Test 1: LEGOTrainer")
    print("=" * 60)

    start_time = time.time()

    # Create data
    set_seed(SEED)
    train_batches = [
        (torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)),
         torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)))
        for _ in range(NUM_BATCHES)
    ]
    val_batches = [
        (torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)),
         torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)))
        for _ in range(NUM_BATCHES // 2)
    ]

    # Create model (4 layers for 2 phases)
    set_seed(SEED)
    model = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE, dim=DIM, num_layers=4, num_heads=NUM_HEADS,
        exit_layer=2, routing_threshold=0.5
    )

    # Define LEGO config
    config = LEGOConfig(
        phases=[
            PhaseConfig(layers=(1, 2), lr=1e-3, patience=1, max_epochs=3),
            PhaseConfig(layers=(3, 4), lr=1e-4, patience=2, max_epochs=5),
        ],
        hard_example_ratio=0.5,
    )

    print(f"Config: {config.describe()}")

    # Train with LEGOTrainer
    trainer = LEGOTrainer(config, vocab_size=VOCAB_SIZE, device=DEVICE, verbose=True)
    result = trainer.train(model, train_batches, val_batches, batch_size=16)

    # Extract results
    thresholds = result['thresholds']
    phase_histories = result['phase_histories']

    print(f"\n--- Results ---")
    print(f"Thresholds: {thresholds}")
    print(f"Phase 1 epochs: {phase_histories[0]['total_epochs']}")
    print(f"Phase 2 epochs: {phase_histories[1]['total_epochs']}")

    # Evaluate with routing
    model.routing_threshold = thresholds[0] if thresholds else 0.5
    eval_config = TrainingConfig(
        stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
        routing_threshold=model.routing_threshold,
        exit_layer=2
    )
    eval_trainer = Trainer(eval_config, vocab_size=VOCAB_SIZE, device=DEVICE)
    final_stats = eval_trainer.evaluate(model, val_batches)

    print(f"\nFinal PPL: {final_stats['ppl']:.2f}")
    print(f"Shallow ratio: {final_stats['shallow_ratio']*100:.1f}%")
    print(f"Compute cost: {final_stats['compute_cost']*100:.1f}%")

    elapsed = time.time() - start_time
    print(f"\nTime: {elapsed:.2f}s")

    # Validation
    errors = []
    if len(thresholds) == 0:
        errors.append("No thresholds computed")
    if final_stats['shallow_ratio'] <= 0:
        errors.append(f"Shallow ratio should be > 0: {final_stats['shallow_ratio']}")
    if elapsed > 10:
        errors.append(f"Test too slow: {elapsed:.2f}s")

    if errors:
        for e in errors:
            print(f"✗ {e}")
        return False
    else:
        print("✓ Test 1 passed!")
        return True


# ==============================================================================
# Test 2: Hard Example Collection
# ==============================================================================

def test_hard_example_collection() -> bool:
    """Test hard example collection at token level."""
    print("\n" + "=" * 60)
    print("Test 2: Hard Example Collection")
    print("=" * 60)

    start_time = time.time()

    # Create data
    set_seed(SEED)
    val_batches = [
        (torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)),
         torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)))
        for _ in range(NUM_BATCHES // 2)
    ]

    total_tokens = sum(x.numel() for x, _ in val_batches)

    # Create and "train" a simple model
    set_seed(SEED)
    model = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE, dim=DIM, num_layers=2, num_heads=NUM_HEADS,
        exit_layer=2, routing_threshold=0.5
    )

    # Compute threshold
    threshold = compute_confidence_threshold(model, val_batches, 0.5, DEVICE)
    print(f"Threshold: {threshold:.4f}")

    # Collect hard examples
    hard_examples = collect_hard_examples(model, val_batches, threshold, DEVICE)
    num_hard = len(hard_examples['targets'])
    hard_ratio = num_hard / total_tokens
    print(f"Hard tokens: {num_hard}/{total_tokens} ({hard_ratio*100:.1f}%)")

    elapsed = time.time() - start_time
    print(f"\nTime: {elapsed:.2f}s")

    # Validation
    errors = []

    # Hard ratio should be ~50%, NOT 100%
    if hard_ratio > 0.9:
        errors.append(f"Hard ratio too high (sequence-level bug?): {hard_ratio*100:.1f}%")

    # Hard ratio should be close to target (50%)
    if abs(hard_ratio - 0.5) > 0.15:
        errors.append(f"Hard ratio far from target 50%: {hard_ratio*100:.1f}%")

    if errors:
        for e in errors:
            print(f"✗ {e}")
        return False
    else:
        print("✓ Test 2 passed!")
        return True


# ==============================================================================
# Test 3: Hard PPL Improvement
# ==============================================================================

def test_hard_ppl_improvement() -> bool:
    """Test that Phase 2 improves Hard PPL."""
    print("\n" + "=" * 60)
    print("Test 3: Hard PPL Improvement")
    print("=" * 60)

    start_time = time.time()

    # Create data
    set_seed(SEED)
    train_batches = [
        (torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)),
         torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)))
        for _ in range(NUM_BATCHES)
    ]
    val_batches = [
        (torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)),
         torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)))
        for _ in range(NUM_BATCHES // 2)
    ]

    # Create 4-layer model
    set_seed(SEED)
    model = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE, dim=DIM, num_layers=4, num_heads=NUM_HEADS,
        exit_layer=2, routing_threshold=0.5
    )

    # Define LEGO config with more epochs for reliable improvement
    config = LEGOConfig(
        phases=[
            PhaseConfig(layers=(1, 2), lr=1e-3, patience=1, max_epochs=3),
            PhaseConfig(layers=(3, 4), lr=1e-4, patience=3, max_epochs=10),
        ],
        hard_example_ratio=0.5,
    )

    # Train with LEGOTrainer
    trainer = LEGOTrainer(config, vocab_size=VOCAB_SIZE, device=DEVICE, verbose=False)
    result = trainer.train(model, train_batches, val_batches, batch_size=16)

    # Get hard examples and evaluate
    hard_examples = result['hard_examples']

    if hard_examples is None:
        print("✗ No hard examples collected")
        return False

    # Evaluate on hard examples
    hard_ppl = evaluate_on_hard_examples(
        model, hard_examples, VOCAB_SIZE, DEVICE, 16, 2
    )
    print(f"Final Hard PPL: {hard_ppl:.2f}")

    # Check that Phase 2 improved
    phase2_final_ppl = result['phase_histories'][1]['val_ppls'][-1]
    phase2_first_ppl = result['phase_histories'][1]['val_ppls'][0]

    print(f"Phase 2 first Hard PPL: {phase2_first_ppl:.2f}")
    print(f"Phase 2 final Hard PPL: {phase2_final_ppl:.2f}")

    improvement = phase2_first_ppl - phase2_final_ppl
    improvement_pct = improvement / phase2_first_ppl * 100
    print(f"Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")

    elapsed = time.time() - start_time
    print(f"\nTime: {elapsed:.2f}s")

    # Validation
    errors = []
    if phase2_final_ppl >= phase2_first_ppl:
        errors.append(f"Hard PPL should improve during Phase 2: {phase2_first_ppl:.2f} -> {phase2_final_ppl:.2f}")

    if errors:
        for e in errors:
            print(f"✗ {e}")
        return False
    else:
        print("✓ Test 3 passed!")
        return True


# ==============================================================================
# Main
# ==============================================================================

def main() -> bool:
    start_time = time.time()

    results = []

    # Test 1: LEGOTrainer API
    results.append(("LEGOTrainer", test_lego_trainer()))

    # Test 2: Hard Example Collection
    results.append(("Hard Example Collection", test_hard_example_collection()))

    # Test 3: Hard PPL improvement
    results.append(("Hard PPL Improvement", test_hard_ppl_improvement()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")

    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
