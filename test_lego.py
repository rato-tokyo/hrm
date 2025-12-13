"""
LEGO Specification Verification Tests

Quick-running tests that verify the specification hasn't changed.
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
    LEGOTransformer,
    Trainer,
    TrainingConfig,
    StageConfig,
    LEGOConfig,
    compute_confidence,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    train_upper_layers,
    evaluate_on_hard_examples,
    create_standard_config,
    create_deep_supervision_config,
)

from helpers import set_seed, create_synthetic_data, assert_close


# ==============================================================================
# Test: LEGOConfig
# ==============================================================================

def test_lego_config():
    """Test LEGOConfig default values."""
    print("\n[TEST] LEGOConfig defaults")

    config = LEGOConfig()

    assert config.phase1_layers == 2, f"phase1_layers: expected 2, got {config.phase1_layers}"
    assert config.phase1_lr == 1e-3, f"phase1_lr: expected 1e-3, got {config.phase1_lr}"
    assert config.phase1_patience == 1, f"phase1_patience: expected 1, got {config.phase1_patience}"
    assert config.hard_example_ratio == 0.5, f"hard_example_ratio: expected 0.5, got {config.hard_example_ratio}"
    assert config.phase2_layers == 4, f"phase2_layers: expected 4, got {config.phase2_layers}"
    assert config.phase2_lr == 1e-4, f"phase2_lr: expected 1e-4, got {config.phase2_lr}"
    assert config.phase2_patience == 3, f"phase2_patience: expected 3, got {config.phase2_patience}"

    print("  All defaults verified")


# ==============================================================================
# Test: TrainingConfig
# ==============================================================================

def test_training_config():
    """Test TrainingConfig and factory functions."""
    print("\n[TEST] TrainingConfig")

    # Standard config
    std_config = create_standard_config(num_layers=3)
    assert len(std_config.stages) == 1
    assert std_config.stages[0].layers == (3, 3)
    assert std_config.stages[0].loss_weight == 1.0
    assert std_config.has_routing == False
    print("  Standard config: OK")

    # Deep supervision config
    ds_config = create_deep_supervision_config(num_layers=3)
    assert len(ds_config.stages) == 3
    for i, stage in enumerate(ds_config.stages):
        assert stage.layers == (i+1, i+1)
        assert_close(stage.loss_weight, 1.0/3, f"stage{i+1}.loss_weight")
    print("  Deep supervision config: OK")

    # Routing config
    routing_config = TrainingConfig(
        stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
        routing_threshold=0.15,
        exit_layer=2
    )
    assert routing_config.has_routing == True
    print("  Routing config: OK")


# ==============================================================================
# Test: LEGOTransformer (standard mode)
# ==============================================================================

def test_lego_transformer_standard():
    """Test LEGOTransformer forward passes (standard mode)."""
    print("\n[TEST] LEGOTransformer (standard)")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)

    # Check architecture
    assert model.num_layers == 2
    assert model.dim == 32
    assert model.vocab_size == 100

    set_seed(42)
    x = torch.randint(0, 100, (2, 8))

    # forward()
    output = model(x)
    assert output.shape == (2, 8, 100)
    expected_mean = 0.0101035647
    assert_close(output.mean().item(), expected_mean, "forward() mean")

    # forward_all_layers()
    outputs = model.forward_all_layers(x)
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 8, 100)
    assert outputs[1].shape == (2, 8, 100)
    # Final layer output should match forward()
    assert torch.allclose(outputs[-1], output, atol=1e-6)
    print("  forward_all_layers() consistency: OK")


# ==============================================================================
# Test: LEGOTransformer (early exit mode)
# ==============================================================================

def test_lego_transformer_early_exit():
    """Test LEGOTransformer forward passes (early exit mode)."""
    print("\n[TEST] LEGOTransformer (early exit)")

    set_seed(42)
    model = LEGOTransformer(
        vocab_size=100, dim=32, num_layers=4, num_heads=2,
        exit_layer=2, routing_threshold=0.5
    )

    # Check architecture
    assert model.num_layers == 4
    assert model.exit_layer == 2
    assert model.routing_threshold == 0.5

    set_seed(42)
    x = torch.randint(0, 100, (2, 8))

    # forward()
    output = model(x)
    assert output.shape == (2, 8, 100)
    expected_mean = -0.0024090516
    assert_close(output.mean().item(), expected_mean, "forward() mean")

    # forward_train()
    train_out = model.forward_train(x)
    assert 'shallow_logits' in train_out
    assert 'deep_logits' in train_out
    assert 'confidence' in train_out
    assert 'shallow_ratio' in train_out

    assert train_out['shallow_logits'].shape == (2, 8, 100)
    assert train_out['deep_logits'].shape == (2, 8, 100)
    assert train_out['confidence'].shape == (2, 8)

    # Deep logits should match forward()
    assert torch.allclose(train_out['deep_logits'], output, atol=1e-6)
    print("  forward_train() consistency: OK")

    # forward_inference()
    inf_output, stats = model.forward_inference(x)
    assert inf_output.shape == (2, 8, 100)
    assert 'mean_confidence' in stats
    assert 'shallow_ratio' in stats
    assert 'compute_cost' in stats
    print("  forward_inference(): OK")


# ==============================================================================
# Test: compute_confidence
# ==============================================================================

def test_compute_confidence():
    """Test compute_confidence function."""
    print("\n[TEST] compute_confidence")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)

    set_seed(42)
    x = torch.randint(0, 100, (2, 8))

    # Get hidden state
    h = model.embedding(x)
    for layer in model.layers:
        h = layer(h)

    # Compute confidence
    confidence = compute_confidence(model, h)

    assert confidence.shape == (2, 8)
    assert (confidence >= 0).all() and (confidence <= 1).all()

    expected_mean = 0.0754802376
    assert_close(confidence.mean().item(), expected_mean, "confidence mean")


# ==============================================================================
# Test: compute_confidence_threshold
# ==============================================================================

def test_compute_confidence_threshold():
    """Test compute_confidence_threshold function."""
    print("\n[TEST] compute_confidence_threshold")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    threshold = compute_confidence_threshold(model, val_batches, target_ratio=0.5, device='cpu')

    expected_threshold = 0.0710195750
    assert_close(threshold, expected_threshold, "threshold")


# ==============================================================================
# Test: collect_hard_examples
# ==============================================================================

def test_collect_hard_examples():
    """Test collect_hard_examples function."""
    print("\n[TEST] collect_hard_examples")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    threshold = 0.0710195750  # From previous test

    hard_examples = collect_hard_examples(model, val_batches, threshold, device='cpu')

    assert 'inputs' in hard_examples
    assert 'hidden_states' in hard_examples
    assert 'targets' in hard_examples
    assert 'confidences' in hard_examples

    num_hard = len(hard_examples['targets'])
    expected_num_hard = 256  # 50% of 4*8*16 = 512
    assert num_hard == expected_num_hard, f"num_hard: expected {expected_num_hard}, got {num_hard}"
    print(f"  num_hard: {num_hard}")

    # Check hidden state shape
    assert hard_examples['hidden_states'].shape == (num_hard, 32)

    # Check all confidences are below threshold
    assert (hard_examples['confidences'] < threshold).all()

    expected_avg_conf = 0.0575758480
    assert_close(hard_examples['confidences'].mean().item(), expected_avg_conf, "avg confidence")


# ==============================================================================
# Test: create_hard_example_loader
# ==============================================================================

def test_create_hard_example_loader():
    """Test create_hard_example_loader function."""
    print("\n[TEST] create_hard_example_loader")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    threshold = 0.0710195750
    hard_examples = collect_hard_examples(model, val_batches, threshold, device='cpu')

    set_seed(42)  # For reproducible shuffling
    hard_batches = create_hard_example_loader(hard_examples, batch_size=64)

    expected_num_batches = 4  # 256 / 64 = 4
    assert len(hard_batches) == expected_num_batches
    print(f"  num_batches: {len(hard_batches)}")

    # Check batch shapes
    h_batch, t_batch = hard_batches[0]
    assert h_batch.shape == (64, 1, 32)  # (batch_size, seq_len=1, dim)
    assert t_batch.shape == (64,)
    print("  batch shapes: OK")


# ==============================================================================
# Test: train_upper_layers
# ==============================================================================

def test_train_upper_layers():
    """Test train_upper_layers function."""
    print("\n[TEST] train_upper_layers")

    # Create Phase 1 model
    set_seed(42)
    model_phase1 = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    threshold = 0.0710195750
    hard_examples = collect_hard_examples(model_phase1, val_batches, threshold, device='cpu')

    # Create extended model using extend_from
    set_seed(42)
    model_extended = LEGOTransformer.extend_from(
        source_model=model_phase1,
        num_layers=4,
        routing_threshold=threshold,
        freeze_lower=True
    )

    # Create hard example loader with fixed seed
    set_seed(42)
    hard_batches = create_hard_example_loader(hard_examples, batch_size=64)

    # Create optimizer (only for trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model_extended.parameters() if p.requires_grad],
        lr=1e-4
    )

    # Train one epoch
    set_seed(42)
    train_loss = train_upper_layers(
        model_extended, hard_batches, optimizer,
        vocab_size=100, device='cpu', num_lower_layers=2
    )

    expected_loss = 5.0222663879
    assert_close(train_loss, expected_loss, "train_loss")


# ==============================================================================
# Test: evaluate_on_hard_examples
# ==============================================================================

def test_evaluate_on_hard_examples():
    """Test evaluate_on_hard_examples function."""
    print("\n[TEST] evaluate_on_hard_examples")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    threshold = 0.0710195750
    hard_examples = collect_hard_examples(model, val_batches, threshold, device='cpu')

    # Evaluate Phase 1 model on hard examples
    ppl = evaluate_on_hard_examples(
        model, hard_examples, vocab_size=100, device='cpu',
        batch_size=64, num_lower_layers=2
    )

    expected_ppl = 160.6869812012
    assert_close(ppl, expected_ppl, "hard_ppl")


# ==============================================================================
# Test: Trainer.compute_loss
# ==============================================================================

def test_trainer_compute_loss():
    """Test Trainer.compute_loss with different configs."""
    print("\n[TEST] Trainer.compute_loss")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)

    set_seed(42)
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))

    # Standard config (final layer only)
    std_config = create_standard_config(num_layers=2)
    trainer_std = Trainer(std_config, vocab_size=100)
    loss_std = trainer_std.compute_loss(model, x, y)

    expected_loss_std = 5.0274624825
    assert_close(loss_std.item(), expected_loss_std, "standard loss")

    # Deep supervision config
    ds_config = create_deep_supervision_config(num_layers=2)
    trainer_ds = Trainer(ds_config, vocab_size=100)
    loss_ds = trainer_ds.compute_loss(model, x, y)

    expected_loss_ds = 4.9431886673
    assert_close(loss_ds.item(), expected_loss_ds, "deep supervision loss")


# ==============================================================================
# Test: Trainer.evaluate (standard)
# ==============================================================================

def test_trainer_evaluate_standard():
    """Test Trainer.evaluate without routing."""
    print("\n[TEST] Trainer.evaluate (standard)")

    set_seed(42)
    model = LEGOTransformer(vocab_size=100, dim=32, num_layers=2, num_heads=2)
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    config = create_standard_config(num_layers=2)
    trainer = Trainer(config, vocab_size=100)

    stats = trainer.evaluate(model, val_batches)

    assert 'ppl' in stats
    assert 'acc' in stats
    assert 'shallow_ratio' in stats
    assert 'compute_cost' in stats

    expected_ppl = 151.8328763710
    expected_acc = 0.0058593750

    assert_close(stats['ppl'], expected_ppl, "ppl")
    assert_close(stats['acc'], expected_acc, "acc")
    assert stats['shallow_ratio'] == 0.0  # No routing
    assert stats['compute_cost'] == 1.0  # Full model


# ==============================================================================
# Test: Trainer.evaluate (routing)
# ==============================================================================

def test_trainer_evaluate_routing():
    """Test Trainer.evaluate with routing."""
    print("\n[TEST] Trainer.evaluate (routing)")

    set_seed(42)
    model = LEGOTransformer(
        vocab_size=100, dim=32, num_layers=4, num_heads=2,
        exit_layer=2, routing_threshold=0.02
    )
    val_batches = create_synthetic_data(num_batches=4, batch_size=8, seq_len=16, vocab_size=100)

    config = TrainingConfig(
        stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
        routing_threshold=0.02,
        exit_layer=2
    )
    trainer = Trainer(config, vocab_size=100)

    stats = trainer.evaluate(model, val_batches)

    expected_ppl = 164.9025456048
    expected_shallow_ratio = 1.0000000000
    expected_compute_cost = 0.5000000000

    assert_close(stats['ppl'], expected_ppl, "ppl")
    assert_close(stats['shallow_ratio'], expected_shallow_ratio, "shallow_ratio")
    assert_close(stats['compute_cost'], expected_compute_cost, "compute_cost")


# ==============================================================================
# Test: Full LEGO Integration (Mini)
# ==============================================================================

def test_lego_integration_mini():
    """Mini integration test for LEGO workflow."""
    print("\n[TEST] LEGO Integration (Mini)")

    # Phase 1: Create and evaluate shallow model
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
    phase1_hard_ppl = evaluate_on_hard_examples(
        model_phase1, hard_examples, vocab_size=100, device='cpu',
        batch_size=64, num_lower_layers=2
    )
    expected_phase1_hard_ppl = 160.6869812012
    assert_close(phase1_hard_ppl, expected_phase1_hard_ppl, "phase1_hard_ppl")

    # Phase 2: Create extended model using extend_from
    set_seed(42)
    model_extended = LEGOTransformer.extend_from(
        source_model=model_phase1,
        num_layers=4,
        routing_threshold=threshold,
        freeze_lower=True
    )

    # Train upper layers
    set_seed(42)
    hard_batches = create_hard_example_loader(hard_examples, batch_size=64)
    optimizer = torch.optim.AdamW(
        [p for p in model_extended.parameters() if p.requires_grad],
        lr=1e-4
    )

    set_seed(42)
    train_loss = train_upper_layers(
        model_extended, hard_batches, optimizer,
        vocab_size=100, device='cpu', num_lower_layers=2
    )
    expected_train_loss = 5.0222663879
    assert_close(train_loss, expected_train_loss, "train_loss")

    # Evaluate Phase 2 on hard examples
    phase2_hard_ppl = evaluate_on_hard_examples(
        model_extended, hard_examples, vocab_size=100, device='cpu',
        batch_size=64, num_lower_layers=2
    )
    expected_phase2_hard_ppl = 146.9902038574
    assert_close(phase2_hard_ppl, expected_phase2_hard_ppl, "phase2_hard_ppl")

    # Final evaluation with routing
    config = TrainingConfig(
        stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
        routing_threshold=threshold,
        exit_layer=2
    )
    trainer = Trainer(config, vocab_size=100)
    stats = trainer.evaluate(model_extended, val_batches)

    expected_final_ppl = 149.0614695912
    expected_shallow_ratio = 0.5000000000
    expected_compute_cost = 0.7500000000

    assert_close(stats['ppl'], expected_final_ppl, "final_ppl")
    assert_close(stats['shallow_ratio'], expected_shallow_ratio, "shallow_ratio")
    assert_close(stats['compute_cost'], expected_compute_cost, "compute_cost")

    print("  Integration test: OK")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("LEGO Specification Verification Tests")
    print("=" * 60)

    tests = [
        test_lego_config,
        test_training_config,
        test_lego_transformer_standard,
        test_lego_transformer_early_exit,
        test_compute_confidence,
        test_compute_confidence_threshold,
        test_collect_hard_examples,
        test_create_hard_example_loader,
        test_train_upper_layers,
        test_evaluate_on_hard_examples,
        test_trainer_compute_loss,
        test_trainer_evaluate_standard,
        test_trainer_evaluate_routing,
        test_lego_integration_mini,
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
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
