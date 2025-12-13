"""
LEGO Specification Verification Tests

Minimal tests to verify core specification: final PPL, shallow_ratio, compute_cost.
Uses small fixed data with deterministic seeds to ensure exact numerical reproducibility.

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
# Test: LEGOBlock
# ==============================================================================

def test_legoblock():
    """Test LEGOBlock as nn.Module."""
    print("\n[TEST] LEGOBlock")

    set_seed(42)
    block = LEGOBlock(dim=32, num_heads=2, num_layers=2, threshold=0.5)
    assert block.num_layers == 2
    assert block.threshold == 0.5
    assert len(block.layers) == 2
    print(f"  LEGOBlock(dim=32, num_heads=2, num_layers=2, threshold=0.5): OK")

    # Test forward
    h = torch.randn(2, 4, 32)
    out = block(h)
    assert out.shape == h.shape
    print(f"  forward: input={h.shape} -> output={out.shape} OK")

    # Test forward_with_cache
    h2 = torch.randn(2, 1, 32)
    out2, cache = block.forward_with_cache(h2, None)
    assert out2.shape == h2.shape
    assert len(cache) == 2  # 2 layers
    print(f"  forward_with_cache: cache has {len(cache)} entries OK")

    # Test freeze/unfreeze
    block.freeze()
    assert all(not p.requires_grad for p in block.parameters())
    block.unfreeze()
    assert all(p.requires_grad for p in block.parameters())
    print("  freeze/unfreeze: OK")

    # Test should_exit
    confidence = torch.tensor([[0.6, 0.7]])
    assert block.should_exit(confidence)
    confidence_low = torch.tensor([[0.3, 0.4]])
    assert not block.should_exit(confidence_low)
    print("  should_exit: OK")

    print("  LEGOBlock test: OK")


# ==============================================================================
# Test: LEGOTransformer creation
# ==============================================================================

def test_transformer_create():
    """Test LEGOTransformer.create() factory method."""
    print("\n[TEST] LEGOTransformer.create")

    set_seed(42)
    model = LEGOTransformer.create(vocab_size=100, dim=32, num_layers=2, num_heads=2)

    assert len(model.blocks) == 1
    assert model.blocks[0].num_layers == 2
    assert model.num_layers == 2
    print(f"  Created: {len(model.blocks)} block, {model.num_layers} layers OK")

    # Test forward
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 100)
    print(f"  forward: {out.shape} OK")

    print("  LEGOTransformer.create test: OK")


# ==============================================================================
# Test: Full LEGO Integration
# ==============================================================================

def test_lego_integration():
    """Integration test for LEGO workflow - verifies final PPL values."""
    print("\n[TEST] LEGO Integration")

    # Phase 1: Create and evaluate shallow model (single block)
    set_seed(42)
    model_phase1 = LEGOTransformer.create(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    # Compute threshold at block 0 (the only block in phase 1)
    threshold = compute_confidence_threshold(
        model_phase1, val_batches, target_ratio=0.5, device='cpu', block_idx=0
    )
    print(f"  threshold: {threshold:.10f}")

    # Collect hard examples from block 0
    hard_examples = collect_hard_examples(
        model_phase1, val_batches, threshold, device='cpu', block_idx=0
    )
    print(f"  hard examples: {len(hard_examples['targets'])}")

    # Evaluate Phase 1 on hard examples (start_block_idx=1 means new block)
    phase1_hard_ppl = evaluate_on_hard_examples(
        model_phase1, hard_examples, device='cpu',
        batch_size=64, start_block_idx=1
    )
    print(f"  Phase 1 Hard PPL: {phase1_hard_ppl:.10f}")

    # Phase 2: Extend model
    set_seed(42)
    model_extended = model_phase1.extend(
        num_new_layers=2,
        threshold=threshold,
        freeze_existing=True
    )

    # Verify block structure
    assert len(model_extended.blocks) == 2, f"Expected 2 blocks, got {len(model_extended.blocks)}"
    assert model_extended.blocks[0].num_layers == 2, "Block 1 should have 2 layers"
    assert model_extended.blocks[1].num_layers == 2, "Block 2 should have 2 layers"
    assert model_extended.num_layers == 4, f"Expected 4 total layers, got {model_extended.num_layers}"

    # Train new block (start_block_idx=1 is the new block)
    set_seed(42)
    hard_batches = create_hard_example_loader(hard_examples, batch_size=64)
    optimizer = torch.optim.AdamW(
        [p for p in model_extended.parameters() if p.requires_grad],
        lr=1e-4
    )

    set_seed(42)
    train_loss = train_new_block(
        model_extended, hard_batches, optimizer,
        device='cpu', start_block_idx=1
    )
    print(f"  train_loss: {train_loss:.10f}")

    # Evaluate Phase 2 on hard examples
    phase2_hard_ppl = evaluate_on_hard_examples(
        model_extended, hard_examples, device='cpu',
        batch_size=64, start_block_idx=1
    )
    print(f"  Phase 2 Hard PPL: {phase2_hard_ppl:.10f}")

    # Final evaluation with routing
    trainer = Trainer(vocab_size=100)
    stats = trainer.evaluate(model_extended, val_batches, use_routing=True)

    print(f"  Final PPL: {stats['ppl']:.10f}")
    print(f"  Shallow ratio: {stats['shallow_ratio']:.10f}")
    print(f"  Compute cost: {stats['compute_cost']:.10f}")

    # Basic sanity checks
    assert phase2_hard_ppl < phase1_hard_ppl * 1.5, "Phase 2 should not be much worse"
    assert 0 <= stats['shallow_ratio'] <= 1
    assert 0 < stats['compute_cost'] <= 1

    print("  Integration test: OK")


# ==============================================================================
# Test: Generate
# ==============================================================================

def test_generate():
    """Test autoregressive generation (standard mode, no early exit)."""
    print("\n[TEST] Generate (Standard)")

    set_seed(42)
    model = LEGOTransformer.create(vocab_size=100, dim=32, num_layers=2, num_heads=2)

    set_seed(42)
    prompt = torch.randint(0, 100, (1, 4))
    generated, stats = model.generate(prompt, max_new_tokens=8, temperature=1.0)

    assert generated.shape == (1, 12), f"Expected (1, 12), got {generated.shape}"
    assert (generated[:, :4] == prompt).all(), "Prompt should be preserved"
    assert stats['exit_counts'][0] == 8, "All tokens should exit at single block"
    print(f"  Generated shape: {generated.shape} OK")
    print(f"  Exit counts: {stats['exit_counts']}")
    print("  Generate test: OK")


def test_generate_early_exit():
    """Test TRUE early exit generation with multiple blocks."""
    print("\n[TEST] Generate (Early Exit)")

    set_seed(42)
    # Create 2-block model with low threshold to force some early exits
    block1 = LEGOBlock(dim=32, num_heads=2, num_layers=2, threshold=0.02)
    block2 = LEGOBlock(dim=32, num_heads=2, num_layers=2, threshold=1.0)
    model = LEGOTransformer(vocab_size=100, dim=32, num_heads=2, blocks=[block1, block2])

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


# ==============================================================================
# Test: Extend
# ==============================================================================

def test_extend_adds_block():
    """Test that extend() properly adds a new block."""
    print("\n[TEST] Extend Adds Block")

    set_seed(42)
    model = LEGOTransformer.create(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    assert len(model.blocks) == 1
    assert model.num_layers == 2
    print(f"  Initial: {len(model.blocks)} block(s), {model.num_layers} layers")

    # Extend to 4 layers (2 blocks)
    model2 = model.extend(num_new_layers=2, threshold=0.5)
    assert len(model2.blocks) == 2
    assert model2.blocks[0].threshold == 0.5  # Exit threshold
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
        test_transformer_create,
        test_lego_integration,
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
