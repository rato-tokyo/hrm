# -*- coding: utf-8 -*-
"""
LEGO Framework - Quick Test Script

ローカルで実行して、LEGOとASHEMが正しく機能しているか確認するテスト。
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
    StandardTransformer,
    DeepSupervisionTransformer,
    Trainer,
    TrainingConfig,
    StageConfig,
    create_standard_config,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    train_upper_layers,
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
# Main Test
# ==============================================================================

def main() -> bool:
    start_time = time.time()

    print("=" * 60)
    print("LEGO Framework - Test Suite")
    print("=" * 60)

    # ==========================================================================
    # Create Data
    # ==========================================================================
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

    total_tokens = sum(x.numel() for x, _ in val_batches)

    # ==========================================================================
    # Phase 1: Train 2-layer model
    # ==========================================================================
    print("\n--- Phase 1: Train 2-layer model ---")
    set_seed(SEED)

    model = StandardTransformer(
        vocab_size=VOCAB_SIZE, dim=DIM, num_layers=2, num_heads=NUM_HEADS
    )
    config = create_standard_config(2)
    trainer = Trainer(config, vocab_size=VOCAB_SIZE, device=DEVICE)
    optimizer = trainer.create_optimizer(model, base_lr=1e-3)

    phase1_start = time.time()
    for epoch in range(3):
        train_loss = trainer.train_epoch(model, train_batches, optimizer)
        val_stats = trainer.evaluate(model, val_batches)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | "
              f"Val PPL={val_stats['ppl']:.2f} | Val Acc={val_stats['acc']*100:.2f}%")
    phase1_time = time.time() - phase1_start

    phase1_stats = trainer.evaluate(model, val_batches)
    print(f"\nPhase 1 Final: PPL={phase1_stats['ppl']:.2f}, Acc={phase1_stats['acc']*100:.2f}%")
    print(f"Phase 1 Time: {phase1_time:.2f}s")

    # ==========================================================================
    # Threshold & Hard Examples
    # ==========================================================================
    print("\n--- Confidence Threshold ---")
    threshold = compute_confidence_threshold(model, val_batches, 0.5, DEVICE)
    print(f"Threshold: {threshold:.4f}")

    print("\n--- Hard Example Collection ---")
    hard_examples = collect_hard_examples(model, val_batches, threshold, DEVICE)
    num_hard = len(hard_examples['targets'])
    hard_ratio = num_hard / total_tokens
    print(f"Hard tokens: {num_hard}/{total_tokens} ({hard_ratio*100:.1f}%)")
    print(f"Avg confidence: {hard_examples['confidences'].mean().item():.4f}")

    # Phase 1 Hard PPL
    phase1_hard_ppl = evaluate_on_hard_examples(
        model, hard_examples, VOCAB_SIZE, DEVICE, 16, 2
    )
    print(f"Phase 1 Hard PPL: {phase1_hard_ppl:.2f}")

    # ==========================================================================
    # Phase 2: Extend to 4 layers, train on hard examples
    # ==========================================================================
    print("\n--- Phase 2: Extend to 4 layers ---")
    set_seed(SEED + 1)

    model_extended = DeepSupervisionTransformer(
        vocab_size=VOCAB_SIZE, dim=DIM, num_layers=4, num_heads=NUM_HEADS,
        exit_layer=2, routing_threshold=threshold
    )

    # Copy weights from Phase 1
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

    trainable = sum(p.numel() for p in model_extended.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_extended.parameters())
    print(f"Trainable params: {trainable}/{total_params} ({trainable/total_params*100:.1f}%)")

    hard_batches = create_hard_example_loader(hard_examples, 16)
    print(f"Hard batches: {len(hard_batches)}")

    optimizer2 = torch.optim.AdamW(
        [p for p in model_extended.parameters() if p.requires_grad],
        lr=1e-4
    )

    phase2_start = time.time()
    for epoch in range(5):
        train_loss = train_upper_layers(
            model_extended, hard_batches, optimizer2, VOCAB_SIZE, DEVICE, 2
        )

        # Eval with routing
        eval_config = TrainingConfig(
            stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
            routing_threshold=threshold,
            exit_layer=2
        )
        eval_trainer = Trainer(eval_config, vocab_size=VOCAB_SIZE, device=DEVICE)
        val_stats = eval_trainer.evaluate(model_extended, val_batches)

        hard_ppl = evaluate_on_hard_examples(
            model_extended, hard_examples, VOCAB_SIZE, DEVICE, 16, 2
        )

        train_ppl = np.exp(train_loss)
        print(f"Epoch {epoch+1}: Train PPL={train_ppl:.2f} | "
              f"Val PPL={val_stats['ppl']:.2f} | Hard PPL={hard_ppl:.2f}")
    phase2_time = time.time() - phase2_start

    phase2_hard_ppl = evaluate_on_hard_examples(
        model_extended, hard_examples, VOCAB_SIZE, DEVICE, 16, 2
    )
    print(f"\nPhase 2 Final Hard PPL: {phase2_hard_ppl:.2f}")
    print(f"Phase 2 Time: {phase2_time:.2f}s")

    # ==========================================================================
    # Final Evaluation (Two-Stage Inference)
    # ==========================================================================
    print("\n--- Final Evaluation (Two-Stage Inference) ---")
    eval_config = TrainingConfig(
        stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
        routing_threshold=threshold,
        exit_layer=2
    )
    eval_trainer = Trainer(eval_config, vocab_size=VOCAB_SIZE, device=DEVICE)
    final_stats = eval_trainer.evaluate(model_extended, val_batches)

    print(f"Final PPL: {final_stats['ppl']:.2f}")
    print(f"Final Acc: {final_stats['acc']*100:.2f}%")
    print(f"Shallow ratio: {final_stats['shallow_ratio']*100:.1f}%")
    print(f"Compute cost: {final_stats['compute_cost']*100:.1f}%")

    # ==========================================================================
    # Summary
    # ==========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Phase 1 PPL: {phase1_stats['ppl']:.2f}")
    print(f"Phase 2 PPL: {final_stats['ppl']:.2f}")
    print(f"PPL change: {final_stats['ppl'] - phase1_stats['ppl']:+.2f}")
    print()
    print(f"Phase 1 Hard PPL: {phase1_hard_ppl:.2f}")
    print(f"Phase 2 Hard PPL: {phase2_hard_ppl:.2f}")
    hard_improvement = phase1_hard_ppl - phase2_hard_ppl
    hard_improvement_pct = hard_improvement / phase1_hard_ppl * 100
    print(f"Hard PPL improvement: {hard_improvement:+.2f} ({hard_improvement_pct:+.1f}%)")
    print()
    print(f"Total time: {total_time:.2f}s")

    # ==========================================================================
    # Validation
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    errors = []

    # 1. Hard PPL should improve
    if phase2_hard_ppl >= phase1_hard_ppl:
        errors.append(
            f"Hard PPL should improve: {phase1_hard_ppl:.2f} -> {phase2_hard_ppl:.2f}"
        )

    # 2. Shallow ratio should be > 0
    if final_stats['shallow_ratio'] <= 0:
        errors.append(
            f"Shallow ratio should be > 0: {final_stats['shallow_ratio']}"
        )

    # 3. Hard ratio should be ~50%, NOT 100% (sequence-level bug)
    if hard_ratio > 0.9:
        errors.append(
            f"Hard ratio too high (sequence-level bug?): {hard_ratio*100:.1f}%"
        )

    # 4. Hard ratio should be close to target (50%)
    if abs(hard_ratio - 0.5) > 0.15:
        errors.append(
            f"Hard ratio far from target 50%: {hard_ratio*100:.1f}%"
        )

    # 5. Time should be reasonable
    if total_time > 10:
        errors.append(
            f"Test too slow: {total_time:.2f}s (should be < 10s)"
        )

    if errors:
        for e in errors:
            print(f"✗ {e}")
        print(f"\n{len(errors)} validation(s) failed!")
        return False
    else:
        print("✓ All validations passed!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
