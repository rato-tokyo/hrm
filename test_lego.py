"""
LEGO Specification Verification Tests

Minimal tests to verify core specification: final PPL, shallow_ratio, compute_cost.
Uses small fixed data with deterministic seeds to ensure exact numerical reproducibility.

IMPORTANT: These expected values are derived from the b02dd4b implementation.
If any test fails, it means the specification has changed - DO NOT modify expected values.

Run: python3 test_lego.py
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')

import torch

from lego import (
    LEGOBlock,
    LEGOTransformer,
    Trainer,
    set_seed,
    create_synthetic_data,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    train_new_block,
    evaluate_on_hard_examples,
)

from helpers import assert_close


# ==============================================================================
# Test: Full LEGO Integration
# ==============================================================================

def test_lego_integration():
    """Integration test for LEGO workflow - verifies final PPL values."""
    print("\n[TEST] LEGO Integration")

    # Phase 1: Create and evaluate shallow model (single block)
    set_seed(42)
    model_phase1 = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    # Compute threshold
    threshold = compute_confidence_threshold(model_phase1, val_batches, target_ratio=0.5, device='cpu')
    expected_threshold = 0.0710195750
    assert_close(threshold, expected_threshold, "threshold")

    # Collect hard examples
    hard_examples = collect_hard_examples(model_phase1, val_batches, threshold, device='cpu')
    expected_num_hard = 256
    assert len(hard_examples['targets']) == expected_num_hard
    print(f"  hard examples: {len(hard_examples['targets'])}")

    # Evaluate Phase 1 on hard examples
    # start_layer=2 means process from layer 2 onwards (Block 1 end)
    phase1_hard_ppl = evaluate_on_hard_examples(
        model_phase1, hard_examples, device='cpu',
        batch_size=64, start_layer=2
    )
    expected_phase1_hard_ppl = 160.6869812012
    assert_close(phase1_hard_ppl, expected_phase1_hard_ppl, "phase1_hard_ppl")

    # Phase 2: Create extended model using extend (adds 2 layers)
    set_seed(42)
    model_extended = model_phase1.extend(
        num_new_layers=2,
        threshold=threshold,
        freeze_existing=True
    )

    # Verify block structure
    assert len(model_extended.blocks) == 2, f"Expected 2 blocks, got {len(model_extended.blocks)}"
    assert model_extended.blocks[0].end_layer == 2, "Block 1 should end at layer 2"
    assert model_extended.blocks[1].end_layer == 4, "Block 2 should end at layer 4"

    # Train new block
    set_seed(42)
    hard_batches = create_hard_example_loader(hard_examples, batch_size=64)
    optimizer = torch.optim.AdamW(
        [p for p in model_extended.parameters() if p.requires_grad],
        lr=1e-4
    )

    set_seed(42)
    # start_layer=2 is where Block 2 starts (Block 1 end_layer)
    train_loss = train_new_block(
        model_extended, hard_batches, optimizer,
        device='cpu', start_layer=2
    )
    expected_train_loss = 5.0222663879
    assert_close(train_loss, expected_train_loss, "train_loss")

    # Evaluate Phase 2 on hard examples
    phase2_hard_ppl = evaluate_on_hard_examples(
        model_extended, hard_examples, device='cpu',
        batch_size=64, start_layer=2
    )
    expected_phase2_hard_ppl = 146.9902038574
    assert_close(phase2_hard_ppl, expected_phase2_hard_ppl, "phase2_hard_ppl")

    # Final evaluation with routing (uses block thresholds)
    trainer = Trainer(vocab_size=100)
    stats = trainer.evaluate(model_extended, val_batches, use_routing=True)

    expected_final_ppl = 149.0614695912
    expected_shallow_ratio = 0.5000000000
    expected_compute_cost = 0.7500000000

    assert_close(stats['ppl'], expected_final_ppl, "final_ppl")
    assert_close(stats['shallow_ratio'], expected_shallow_ratio, "shallow_ratio")
    assert_close(stats['compute_cost'], expected_compute_cost, "compute_cost")

    print("  Phase 1 Hard PPL: {:.4f}".format(phase1_hard_ppl))
    print("  Phase 2 Hard PPL: {:.4f}".format(phase2_hard_ppl))
    print("  Final PPL: {:.4f}".format(stats['ppl']))
    print("  Shallow ratio: {:.2%}".format(stats['shallow_ratio']))
    print("  Compute cost: {:.2%}".format(stats['compute_cost']))
    print("  Integration test: OK")


# ==============================================================================
# Test: KV Cache
# ==============================================================================

def test_kv_cache():
    """Test KV cache produces same output as non-cached forward."""
    print("\n[TEST] KV Cache")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    model.eval()

    # Create test input
    set_seed(42)
    input_ids = torch.randint(0, 100, (2, 8))  # batch=2, seq=8

    with torch.no_grad():
        # Standard forward (no cache)
        logits_no_cache = model.forward_with_cache(input_ids, use_cache=False)

        # Forward with cache (full sequence)
        logits_with_cache, kv_cache = model.forward_with_cache(input_ids, use_cache=True)

        # Verify same output
        diff_full = (logits_no_cache - logits_with_cache).abs().max().item()
        assert diff_full < 1e-5, f"Full sequence cache mismatch: {diff_full}"
        print(f"  Full sequence cache: diff={diff_full:.2e} OK")

        # Test incremental decoding
        # Process first 4 tokens
        logits_prefix, cache_prefix = model.forward_with_cache(
            input_ids[:, :4], use_cache=True
        )

        # Process remaining 4 tokens with cache
        logits_suffix, _ = model.forward_with_cache(
            input_ids[:, 4:], past_kv_cache=cache_prefix, use_cache=True
        )

        # Compare with full forward
        diff_suffix = (logits_with_cache[:, 4:, :] - logits_suffix).abs().max().item()
        assert diff_suffix < 1e-5, f"Incremental cache mismatch: {diff_suffix}"
        print(f"  Incremental cache: diff={diff_suffix:.2e} OK")

    print("  KV Cache test: OK")


def test_generate():
    """Test autoregressive generation (standard mode, no early exit)."""
    print("\n[TEST] Generate (Standard)")

    set_seed(42)
    # Single block model (no early exit possible)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)

    set_seed(42)
    prompt = torch.randint(0, 100, (1, 4))  # batch=1, seq=4
    generated, stats = model.generate(prompt, max_new_tokens=8, temperature=1.0)

    assert generated.shape == (1, 12), f"Expected (1, 12), got {generated.shape}"
    assert (generated[:, :4] == prompt).all(), "Prompt should be preserved"
    # Single block: all tokens exit at block 0
    assert stats['exit_counts'][0] == 8, "All tokens should exit at single block"
    print(f"  Generated shape: {generated.shape} OK")
    print(f"  Generated tokens: {generated[0].tolist()}")
    print(f"  Exit counts: {stats['exit_counts']}")
    print("  Generate test: OK")


def test_generate_early_exit():
    """Test TRUE early exit generation with LEGOBlock."""
    print("\n[TEST] Generate (Early Exit)")

    set_seed(42)
    # 4-layer model with 2 blocks: Block 1 (layers 0-1), Block 2 (layers 2-3)
    # Use very low threshold (0.02) to ensure some early exits with untrained model
    blocks = [
        LEGOBlock(0, 2, threshold=0.02),  # Block 1 with low threshold
        LEGOBlock(2, 4, threshold=1.0),   # Block 2 (final)
    ]
    model = LEGOTransformer(
        vocab_size=100, dim=32, num_layers=4, num_heads=2,
        blocks=blocks
    )

    set_seed(42)
    prompt = torch.randint(0, 100, (1, 4))

    generated, stats = model.generate(prompt, max_new_tokens=16)

    assert generated.shape == (1, 20), f"Expected (1, 20), got {generated.shape}"
    assert (generated[:, :4] == prompt).all(), "Prompt should be preserved"

    print(f"  Generated shape: {generated.shape} OK")
    print(f"  Exit counts: {stats['exit_counts']}")
    print(f"  Shallow ratio: {stats['shallow_ratio']:.2%}")
    print(f"  Actual compute cost: {stats['actual_compute_cost']:.2%}")

    # Verify compute cost is less than 100% if any shallow exits occurred
    if stats['exit_counts'][0] > 0:
        assert stats['actual_compute_cost'] < 1.0, "Compute cost should be < 100% with early exits"
        print("  Compute cost reduction verified: OK")

    print("  Generate with Early Exit test: OK")


def test_legoblock():
    """Test LEGOBlock dataclass."""
    print("\n[TEST] LEGOBlock")

    block = LEGOBlock(0, 2, threshold=0.5)
    assert block.start_layer == 0
    assert block.end_layer == 2
    assert block.threshold == 0.5
    assert block.num_layers == 2
    print(f"  LEGOBlock(0, 2, 0.5): num_layers={block.num_layers} OK")

    # Default threshold
    block_default = LEGOBlock(2, 4)
    assert block_default.threshold == 1.0
    print(f"  LEGOBlock(2, 4): default threshold={block_default.threshold} OK")

    print("  LEGOBlock test: OK")


def test_extend_adds_block():
    """Test that extend() properly adds a new block."""
    print("\n[TEST] Extend Adds Block")

    set_seed(42)
    # Start with 2-layer model (1 block)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    assert len(model.blocks) == 1
    assert model.blocks[0].end_layer == 2
    print(f"  Initial: {len(model.blocks)} block(s), {model.num_layers} layers")

    # Extend to 4 layers (2 blocks)
    model2 = model.extend(num_new_layers=2, threshold=0.5)
    assert len(model2.blocks) == 2
    assert model2.blocks[0].end_layer == 2
    assert model2.blocks[0].threshold == 0.5  # Exit threshold
    assert model2.blocks[1].end_layer == 4
    assert model2.blocks[1].threshold == 1.0  # Final block
    assert model2.num_layers == 4
    print(f"  After extend(2): {len(model2.blocks)} blocks, {model2.num_layers} layers")

    # Extend to 6 layers (3 blocks)
    model3 = model2.extend(num_new_layers=2, threshold=0.7)
    assert len(model3.blocks) == 3
    assert model3.blocks[0].threshold == 0.5  # Preserved
    assert model3.blocks[1].threshold == 0.7  # New exit point
    assert model3.blocks[2].threshold == 1.0  # Final block
    assert model3.num_layers == 6
    print(f"  After extend(2): {len(model3.blocks)} blocks, {model3.num_layers} layers")

    print("  Extend test: OK")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("LEGO Specification Verification Tests")
    print("=" * 60)

    tests = [
        test_legoblock,
        test_lego_integration,
        test_kv_cache,
        test_generate,
        test_generate_early_exit,
        test_extend_adds_block,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
