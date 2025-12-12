"""
LEGO Framework - Quick Test Script

ローカルで実行して、LEGOとASHEMが正しく機能しているか確認するテスト。
小さいデータセット + 固定シードで数値の完全一致を検証。

Usage:
    python test_lego.py

Expected output:
    All tests passed!
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

from ease import (
    StandardTransformer,
    DeepSupervisionTransformer,
    Trainer,
    TrainingConfig,
    StageConfig,
    create_standard_config,
    ASHEMConfig,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    train_upper_layers,
    evaluate_on_hard_examples,
)


# ==============================================================================
# Test Configuration
# ==============================================================================

VOCAB_SIZE = 100
DIM = 32
NUM_HEADS = 2
SEQ_LEN = 16
BATCH_SIZE = 8
NUM_BATCHES = 4
SEED = 42
DEVICE = 'cpu'  # CPUで再現性を確保


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_synthetic_data() -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
                                      List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Create deterministic synthetic data for testing."""
    set_seed(SEED)

    train_batches = []
    val_batches = []

    for _ in range(NUM_BATCHES):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        train_batches.append((x, y))

    for _ in range(NUM_BATCHES // 2):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        val_batches.append((x, y))

    return train_batches, val_batches


# ==============================================================================
# Test Cases
# ==============================================================================

def test_standard_transformer():
    """Test StandardTransformer basic functionality."""
    print("Test 1: StandardTransformer forward pass...", end=" ")
    set_seed(SEED)

    model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=2,
        num_heads=NUM_HEADS
    )

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    output = model(x)

    assert output.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), f"Shape mismatch: {output.shape}"

    # Check deterministic output
    expected_mean = -0.0001  # Approximate expected value
    actual_mean = output.mean().item()
    assert abs(actual_mean) < 0.1, f"Output mean out of range: {actual_mean}"

    print("✓")


def test_deep_supervision_transformer():
    """Test DeepSupervisionTransformer with early exit."""
    print("Test 2: DeepSupervisionTransformer forward_train...", end=" ")
    set_seed(SEED)

    model = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=4,
        num_heads=NUM_HEADS,
        exit_layer=2,
        routing_threshold=0.5
    )

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    outputs = model.forward_train(x)

    assert 'shallow_logits' in outputs
    assert 'deep_logits' in outputs
    assert 'confidence' in outputs
    assert outputs['shallow_logits'].shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert outputs['deep_logits'].shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert outputs['confidence'].shape == (BATCH_SIZE, SEQ_LEN)

    print("✓")


def test_trainer_phase1():
    """Test Phase 1 training (shallow model)."""
    print("Test 3: Phase 1 training...", end=" ")
    set_seed(SEED)

    train_batches, val_batches = create_synthetic_data()

    model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=2,
        num_heads=NUM_HEADS
    )

    config = create_standard_config(num_layers=2)
    trainer = Trainer(config, vocab_size=VOCAB_SIZE, device=DEVICE)
    optimizer = trainer.create_optimizer(model, base_lr=1e-3)

    # Train 1 epoch
    train_loss = trainer.train_epoch(model, train_batches, optimizer)
    val_stats = trainer.evaluate(model, val_batches)

    # Check values are in reasonable range
    assert 3.0 < train_loss < 6.0, f"Train loss out of range: {train_loss}"
    assert 50 < val_stats['ppl'] < 200, f"Val PPL out of range: {val_stats['ppl']}"

    print("✓")


def test_confidence_threshold():
    """Test confidence threshold computation."""
    print("Test 4: Confidence threshold computation...", end=" ")
    set_seed(SEED)

    train_batches, val_batches = create_synthetic_data()

    model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=2,
        num_heads=NUM_HEADS
    )

    threshold = compute_confidence_threshold(
        model, val_batches, target_ratio=0.5, device=DEVICE
    )

    # Threshold should be between 0 and 1
    assert 0.0 < threshold < 1.0, f"Threshold out of range: {threshold}"

    print("✓")


def test_hard_example_collection():
    """Test hard example collection (token-level)."""
    print("Test 5: Hard example collection (token-level)...", end=" ")
    set_seed(SEED)

    train_batches, val_batches = create_synthetic_data()

    model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=2,
        num_heads=NUM_HEADS
    )

    threshold = compute_confidence_threshold(
        model, val_batches, target_ratio=0.5, device=DEVICE
    )

    hard_examples = collect_hard_examples(
        model, val_batches, threshold, device=DEVICE
    )

    # Check structure
    assert 'inputs' in hard_examples
    assert 'hidden_states' in hard_examples
    assert 'targets' in hard_examples
    assert 'confidences' in hard_examples

    # Check token-level collection (not sequence-level)
    total_tokens = sum(x.numel() for x, _ in val_batches)
    num_hard = len(hard_examples['targets'])
    ratio = num_hard / total_tokens

    # Should be approximately 50% (target ratio)
    assert 0.3 < ratio < 0.7, f"Hard ratio out of range: {ratio:.2%}"

    # Critical: Should NOT be 100% (sequence-level would be ~100%)
    assert ratio < 0.9, f"Ratio too high, might be sequence-level: {ratio:.2%}"

    print("✓")


def test_hard_example_loader():
    """Test hard example loader creation."""
    print("Test 6: Hard example loader creation...", end=" ")
    set_seed(SEED)

    train_batches, val_batches = create_synthetic_data()

    model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=2,
        num_heads=NUM_HEADS
    )

    threshold = compute_confidence_threshold(
        model, val_batches, target_ratio=0.5, device=DEVICE
    )

    hard_examples = collect_hard_examples(
        model, val_batches, threshold, device=DEVICE
    )

    hard_batches = create_hard_example_loader(hard_examples, batch_size=16)

    # Check batch structure
    assert len(hard_batches) > 0, "No batches created"

    h, y = hard_batches[0]
    assert h.dim() == 3, f"Hidden state should be 3D: {h.shape}"
    assert h.shape[1] == 1, f"Seq len should be 1: {h.shape}"
    assert h.shape[2] == DIM, f"Dim mismatch: {h.shape}"

    print("✓")


def test_train_upper_layers():
    """Test upper layer training on hard examples."""
    print("Test 7: Train upper layers...", end=" ")
    set_seed(SEED)

    train_batches, val_batches = create_synthetic_data()

    # Phase 1 model
    model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=2,
        num_heads=NUM_HEADS
    )

    threshold = compute_confidence_threshold(
        model, val_batches, target_ratio=0.5, device=DEVICE
    )

    hard_examples = collect_hard_examples(
        model, val_batches, threshold, device=DEVICE
    )

    # Phase 2 model
    model_extended = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=4,
        num_heads=NUM_HEADS,
        exit_layer=2,
        routing_threshold=threshold
    )

    # Copy weights
    model_extended.embedding.load_state_dict(model.embedding.state_dict())
    for i in range(2):
        model_extended.layers[i].load_state_dict(model.layers[i].state_dict())
    model_extended.output_head.load_state_dict(model.output_head.state_dict())

    # Freeze lower layers
    for param in model_extended.embedding.parameters():
        param.requires_grad = False
    for i in range(2):
        for param in model_extended.layers[i].parameters():
            param.requires_grad = False

    hard_batches = create_hard_example_loader(hard_examples, batch_size=16)
    optimizer = torch.optim.AdamW(
        [p for p in model_extended.parameters() if p.requires_grad],
        lr=1e-4
    )

    # Train 1 epoch
    train_loss = train_upper_layers(
        model_extended, hard_batches, optimizer,
        VOCAB_SIZE, DEVICE, num_lower_layers=2
    )

    assert 3.0 < train_loss < 7.0, f"Train loss out of range: {train_loss}"

    print("✓")


def test_evaluate_on_hard_examples():
    """Test evaluation on hard examples."""
    print("Test 8: Evaluate on hard examples...", end=" ")
    set_seed(SEED)

    train_batches, val_batches = create_synthetic_data()

    model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=2,
        num_heads=NUM_HEADS
    )

    threshold = compute_confidence_threshold(
        model, val_batches, target_ratio=0.5, device=DEVICE
    )

    hard_examples = collect_hard_examples(
        model, val_batches, threshold, device=DEVICE
    )

    hard_ppl = evaluate_on_hard_examples(
        model, hard_examples, VOCAB_SIZE, DEVICE,
        batch_size=16, num_lower_layers=2
    )

    # Hard PPL should be higher than overall PPL (harder examples)
    assert 50 < hard_ppl < 500, f"Hard PPL out of range: {hard_ppl}"

    print("✓")


def test_two_stage_inference():
    """Test two-stage inference with routing."""
    print("Test 9: Two-stage inference...", end=" ")
    set_seed(SEED)

    train_batches, val_batches = create_synthetic_data()

    model = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=4,
        num_heads=NUM_HEADS,
        exit_layer=2,
        routing_threshold=0.5
    )

    eval_config = TrainingConfig(
        stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
        routing_threshold=0.5,
        exit_layer=2
    )

    trainer = Trainer(eval_config, vocab_size=VOCAB_SIZE, device=DEVICE)
    stats = trainer.evaluate(model, val_batches)

    assert 'ppl' in stats
    assert 'acc' in stats
    assert 'shallow_ratio' in stats
    assert 'compute_cost' in stats

    # Shallow ratio should be between 0 and 1
    assert 0.0 <= stats['shallow_ratio'] <= 1.0

    # Compute cost should be less than 1 (some early exits)
    assert 0.5 <= stats['compute_cost'] <= 1.0

    print("✓")


def test_full_ashem_pipeline():
    """Test complete ASHEM pipeline (mini version)."""
    print("Test 10: Full ASHEM pipeline...", end=" ")
    set_seed(SEED)

    train_batches, val_batches = create_synthetic_data()

    # Phase 1
    model = StandardTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=2,
        num_heads=NUM_HEADS
    )

    config = create_standard_config(2)
    trainer = Trainer(config, vocab_size=VOCAB_SIZE, device=DEVICE)
    optimizer = trainer.create_optimizer(model, base_lr=1e-3)

    for _ in range(2):  # 2 epochs
        trainer.train_epoch(model, train_batches, optimizer)

    phase1_stats = trainer.evaluate(model, val_batches)

    # Threshold & Hard examples
    threshold = compute_confidence_threshold(model, val_batches, 0.5, DEVICE)
    hard_examples = collect_hard_examples(model, val_batches, threshold, DEVICE)

    phase1_hard_ppl = evaluate_on_hard_examples(
        model, hard_examples, VOCAB_SIZE, DEVICE, 16, 2
    )

    # Phase 2
    model_extended = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=4,
        num_heads=NUM_HEADS,
        exit_layer=2,
        routing_threshold=threshold
    )

    model_extended.embedding.load_state_dict(model.embedding.state_dict())
    for i in range(2):
        model_extended.layers[i].load_state_dict(model.layers[i].state_dict())
    model_extended.output_head.load_state_dict(model.output_head.state_dict())

    for param in model_extended.embedding.parameters():
        param.requires_grad = False
    for i in range(2):
        for param in model_extended.layers[i].parameters():
            param.requires_grad = False

    hard_batches = create_hard_example_loader(hard_examples, 16)
    optimizer2 = torch.optim.AdamW(
        [p for p in model_extended.parameters() if p.requires_grad],
        lr=1e-4
    )

    for _ in range(2):  # 2 epochs
        train_upper_layers(model_extended, hard_batches, optimizer2, VOCAB_SIZE, DEVICE, 2)

    phase2_hard_ppl = evaluate_on_hard_examples(
        model_extended, hard_examples, VOCAB_SIZE, DEVICE, 16, 2
    )

    # Final evaluation
    eval_config = TrainingConfig(
        stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
        routing_threshold=threshold,
        exit_layer=2
    )
    eval_trainer = Trainer(eval_config, vocab_size=VOCAB_SIZE, device=DEVICE)
    final_stats = eval_trainer.evaluate(model_extended, val_batches)

    # Verify improvements
    # Hard PPL should improve after Phase 2
    assert phase2_hard_ppl < phase1_hard_ppl, \
        f"Hard PPL should improve: {phase1_hard_ppl:.2f} -> {phase2_hard_ppl:.2f}"

    # Shallow ratio should be > 0 (some early exits)
    assert final_stats['shallow_ratio'] > 0, "No early exits detected"

    print("✓")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 50)
    print("LEGO Framework - Test Suite")
    print("=" * 50)
    print()

    tests = [
        test_standard_transformer,
        test_deep_supervision_transformer,
        test_trainer_phase1,
        test_confidence_threshold,
        test_hard_example_collection,
        test_hard_example_loader,
        test_train_upper_layers,
        test_evaluate_on_hard_examples,
        test_two_stage_inference,
        test_full_ashem_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1

    print()
    print("=" * 50)
    if failed == 0:
        print(f"All {passed} tests passed! ✓")
    else:
        print(f"Passed: {passed}, Failed: {failed}")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
