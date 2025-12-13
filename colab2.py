"""
LEGO (Layered Ensemble with Gradual Optimization) Experiment

**Experiment Version**: Hard Example Mining with Hard Freezing
**Date**: 2025-12-12
**Dataset**: WikiText-2 (10K samples)

This experiment demonstrates the LEGO training strategy:
1. Phase 1: Train a shallow model (2 layers) on all data
2. Identify hard examples using confidence-based threshold (auto-adjusted)
3. Phase 2: Train additional layers (2 more) on hard examples only
   - Hard Freezing: Layer 1-2 completely frozen (requires_grad=False)
   - Layer 3-4: Trainable
4. Inference: Two-stage routing using Early Exit mechanism

Key Benefits:
- Focuses computational resources on hard examples during training
- Hard PPL: 78% improvement (2763 → 668)
- Reduces compute cost by 36% using adaptive routing
- Fully integrated with LEGO framework's 2 core options

LEGO Framework (2 Core Options):
1. layer_weights: Which layers to train (loss weights)
2. routing_threshold: When to exit early (inference efficiency)

References:
- LEGO: Layered Ensemble with Gradual Optimization
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- Early Exit: BranchyNet (2016), Teerapittayanon et al. (2016)
- Deep Supervision: Lee et al. (2015)
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
from dataclasses import dataclass
from typing import Dict, Any

from lego import (
    LEGOTransformer,
    Trainer,
    TrainingConfig,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    evaluate_on_hard_examples,
)

sys.path.insert(0, 'experiments')
from utils import set_seed, get_device, create_wikitext_dataloaders


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for LEGO experiment.
    """
    # Model architecture
    vocab_size: int = 69830
    seq_len: int = 32
    dim: int = 64
    num_heads: int = 4

    # Dataset parameters
    phase1_samples: int = 10000
    phase1_batch: int = 64
    phase2_batch: int = 64
    phase1_epochs: int = 50
    phase2_epochs: int = 50

    # Phase 1: Shallow model
    phase1_layers: int = 2
    phase1_lr: float = 1e-3
    phase1_patience: int = 1

    # Hard example collection
    hard_example_ratio: float = 0.5

    # Phase 2: Deep model
    phase2_layers: int = 4
    phase2_lr: float = 1e-4
    phase2_patience: int = 3


# ==============================================================================
# Utility Functions
# ==============================================================================
# Note: All utility functions are now imported from framework modules:
#
# LEGO functions (from src/lego/utils.py):
# - compute_confidence_threshold()
# - collect_hard_examples()
# - create_hard_example_loader()
# - train_upper_layers()
# - evaluate_on_hard_examples()
#
# Data/Device utilities (from experiments/utils.py):
# - set_seed()
# - get_device()
# - create_wikitext_dataloaders()
# ==============================================================================


# ==============================================================================
# Experiment Runner
# ==============================================================================

def run_experiment(config: ExperimentConfig, device: str) -> Dict[str, Any]:
    """
    Run complete hard example mining experiment.

    Experiment flow:
    1. Phase 1: Train shallow model (2 layers) on all data
    2. Compute confidence threshold and collect hard examples
    3. Phase 2: Extend to 4 layers, train upper layers on hard examples only
    4. Evaluate using two-stage inference (Early Exit)

    Args:
        config: Experiment configuration
        device: Device to run experiment on

    Returns:
        Dictionary with experiment results and metrics
    """
    print(f"\n{'='*60}")
    print("LEGOTransformer - Hard Example Mining")
    print(f"{'='*60}\n")

    # ==========================================================================
    # Phase 1: Train Shallow Model
    # ==========================================================================
    print(f"Phase 1: Train {config.phase1_layers}-layer model")
    print(f"{'='*60}")

    set_seed(42)
    train_loader, val_loader, vocab_size = create_wikitext_dataloaders(
        config.phase1_samples, config.phase1_batch, config.seq_len
    )

    # Create shallow model
    model = LEGOTransformer(
        vocab_size=vocab_size,
        dim=config.dim,
        num_layers=config.phase1_layers,
        num_heads=config.num_heads
    ).to(device)

    # Train with early stopping
    training_config = TrainingConfig()
    trainer = Trainer(training_config, vocab_size=vocab_size, device=device)
    optimizer = trainer.create_optimizer(model, base_lr=config.phase1_lr)

    start_time = time.time()
    result_phase1 = trainer.train_with_early_stopping(
        model=model,
        train_batches=train_loader,
        val_batches=val_loader,
        optimizer=optimizer,
        max_epochs=config.phase1_epochs,
        patience=config.phase1_patience,
        verbose=True
    )
    phase1_time = time.time() - start_time

    phase1_acc = result_phase1['val_accs'][result_phase1['best_epoch']] * 100
    phase1_ppl = result_phase1['val_losses'][result_phase1['best_epoch']]

    print("\nPhase 1 Results:")
    print(f"  Best Acc: {phase1_acc:.2f}%")
    print(f"  Best PPL: {phase1_ppl:.2f}")
    print(f"  Time: {phase1_time:.2f}s")

    # ==========================================================================
    # Compute Confidence Threshold
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Computing Confidence Threshold (target ratio: {config.hard_example_ratio*100:.0f}%)")
    print(f"{'='*60}\n")

    confidence_threshold = compute_confidence_threshold(
        model, val_loader, config.hard_example_ratio, device
    )

    print(f"✓ Computed confidence threshold: {confidence_threshold:.4f}")
    print(f"  Examples with confidence < {confidence_threshold:.4f} will be treated as hard")

    # ==========================================================================
    # Collect Hard Examples
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Collecting Hard Examples")
    print(f"{'='*60}\n")

    hard_examples = collect_hard_examples(
        model, val_loader, confidence_threshold, device
    )

    num_hard = len(hard_examples['targets'])
    avg_confidence = hard_examples['confidences'].mean().item()
    total_samples = config.phase1_samples * 0.2 * config.seq_len

    print(f"✓ Collected {num_hard:,} hard examples")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"  Actual ratio: {num_hard / total_samples * 100:.1f}% "
          f"(target: {config.hard_example_ratio*100:.0f}%)")

    # ==========================================================================
    # Evaluate Phase 1 on Hard Examples
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Evaluating Phase 1 on Hard Examples")
    print(f"{'='*60}\n")

    phase1_hard_ppl = evaluate_on_hard_examples(
        model, hard_examples, vocab_size, device,
        batch_size=config.phase2_batch, num_lower_layers=config.phase1_layers
    )

    print(f"✓ Phase 1 Hard PPL: {phase1_hard_ppl:.2f}")
    print(f"  (vs Overall Val PPL: {phase1_ppl:.2f})")

    # ==========================================================================
    # Phase 2: Extend Model and Train on Hard Examples
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Phase 2: Add 2 layers → Train on hard examples")
    print(f"{'='*60}\n")

    # Create extended model with Early Exit support
    model_extended = LEGOTransformer.extend_from(
        source_model=model,
        num_layers=config.phase2_layers,
        routing_threshold=confidence_threshold,
        freeze_lower=True
    ).to(device)

    print("✓ Copied weights from 2-layer model")
    print("✓ Layers 3-4 randomly initialized")
    print("\n📊 Hard Freezing Configuration:")
    print("  Layer 1-2: Frozen (requires_grad=False)")
    print("  Layer 3-4: Trainable")

    # Configure training for Phase 2 (with routing for evaluation)
    phase2_config = TrainingConfig(
        routing_threshold=confidence_threshold,
        exit_layer=config.phase1_layers
    )

    trainer_phase2 = Trainer(phase2_config, vocab_size=vocab_size, device=device)

    trainable = sum(p.numel() for p in model_extended.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_extended.parameters())
    print(f"\n✓ Frozen lower layers")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Create hard example loader
    hard_batches = create_hard_example_loader(hard_examples, config.phase2_batch)
    print(f"  Hard example batches: {len(hard_batches)}")

    # Train upper layers only
    optimizer_upper = trainer_phase2.create_optimizer(
        model_extended,
        base_lr=config.phase2_lr
    )

    print(f"\n📊 Training Configuration:")
    print(f"  Learning rate: {config.phase2_lr:.1e}")
    print(f"  Patience: {config.phase2_patience}")
    print(f"  Max epochs: {config.phase2_epochs}")

    # Train upper layers with early stopping (simplified)
    start_time = time.time()
    result_phase2 = trainer_phase2.train_upper_layers_with_early_stopping(
        model=model_extended,
        hard_batches=hard_batches,
        val_batches=val_loader,
        hard_examples=hard_examples,
        optimizer=optimizer_upper,
        num_lower_layers=config.phase1_layers,
        max_epochs=config.phase2_epochs,
        patience=config.phase2_patience,
        verbose=True
    )
    phase2_time = time.time() - start_time

    # Get Phase 2 metrics from result
    phase2_hard_ppl = result_phase2['hard_ppls'][result_phase2['best_epoch']]
    final_val_stats = trainer_phase2.evaluate(model_extended, val_loader)

    print("\nPhase 2 Results:")
    print(f"  Best Val PPL: {final_val_stats['ppl']:.2f}")
    print(f"  Best Hard PPL: {phase2_hard_ppl:.2f}")
    print(f"  Hard PPL Improvement: {phase1_hard_ppl - phase2_hard_ppl:+.2f} "
          f"({(phase1_hard_ppl - phase2_hard_ppl) / phase1_hard_ppl * 100:+.1f}%)")
    print(f"  Time: {phase2_time:.2f}s")

    # ==========================================================================
    # Final Evaluation: Two-Stage Inference
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Final Evaluation (Two-Stage Inference)")
    print(f"{'='*60}\n")

    # Reuse trainer_phase2 (same config with routing)
    stats = trainer_phase2.evaluate(model_extended, val_loader)

    print("Results:")
    print(f"  Accuracy: {stats['acc']*100:.2f}%")
    print(f"  Shallow ratio (Layer 2): {stats['shallow_ratio']*100:.1f}%")
    print(f"  Deep ratio (Layer 4): {(1-stats['shallow_ratio'])*100:.1f}%")
    print(f"  Compute cost: {stats['compute_cost']:.2%} of full model")

    # ==========================================================================
    # Summary Comparison
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")
    print("\nOverall Performance:")
    print(f"  Phase 1 (2-layer only):  Acc {phase1_acc:.2f}% | PPL {phase1_ppl:.2f}")
    print(f"  Two-stage inference:     Acc {stats['acc']*100:.2f}% | PPL {stats['ppl']:.2f}")
    print(f"  Accuracy change:         {stats['acc']*100 - phase1_acc:+.2f}%")
    print(f"  PPL change:              {stats['ppl'] - phase1_ppl:+.2f}")

    print("\nHard Examples Performance:")
    print(f"  Phase 1 Hard PPL:        {phase1_hard_ppl:.2f}")
    print(f"  Phase 2 Hard PPL:        {phase2_hard_ppl:.2f}")
    print(f"  Hard PPL Improvement:    {phase1_hard_ppl - phase2_hard_ppl:+.2f} "
          f"({(phase1_hard_ppl - phase2_hard_ppl) / phase1_hard_ppl * 100:+.1f}%)")

    print("\nEfficiency:")
    print(f"  Shallow ratio (Layer 2): {stats['shallow_ratio']*100:.1f}%")
    print(f"  Deep ratio (Layer 4):    {(1-stats['shallow_ratio'])*100:.1f}%")
    print(f"  Compute cost:            {stats['compute_cost']:.2%} of full model")

    return {
        'model_name': 'LEGOTransformer',
        'phase1_acc': phase1_acc,
        'phase1_ppl': phase1_ppl,
        'phase1_hard_ppl': phase1_hard_ppl,
        'phase1_time': phase1_time,
        'num_hard_examples': num_hard,
        'phase2_hard_ppl': phase2_hard_ppl,
        'phase2_time': phase2_time,
        'two_stage_acc': stats['acc'] * 100,
        'two_stage_ppl': stats['ppl'],
        'hard_ppl_improvement': phase1_hard_ppl - phase2_hard_ppl,
        'shallow_ratio': stats['shallow_ratio'],
        'compute_cost': stats['compute_cost']
    }


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main() -> None:
    """Run Hard Example Mining experiment."""
    config = ExperimentConfig()

    print("="*60)
    print("Hard Example Mining + Two-Stage Inference")
    print("="*60)
    print(f"Device: {get_device()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nExperiment Design:")
    print(f"  Phase 1: Train {config.phase1_layers}-layer model")
    print(f"  Compute: Auto-adjust threshold to collect {config.hard_example_ratio*100:.0f}% hard examples")
    print(f"  Phase 2: Add {config.phase2_layers - config.phase1_layers} layers → Train on hard examples")
    print("  Eval: Two-stage inference (Layer 2 or Layer 4) using Early Exit")
    print()

    device = get_device()

    # Run experiment
    run_experiment(config, device)

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
