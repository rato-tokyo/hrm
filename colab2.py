"""
ASHEM (Adaptive Supervision via Hard Example Mining) Experiment

**Experiment Version**: ASHEM with Hard Freezing
**Date**: 2025-12-12
**Dataset**: WikiText-2 (10K samples)

This experiment demonstrates the ASHEM training strategy integrated with LEGO framework:
1. Phase 1: Train a shallow model (2 layers) on all data
2. Identify hard examples using confidence-based threshold (auto-adjusted)
3. Phase 2: Train additional layers (2 more) on hard examples only
   - Hard Freezing: Layer 1-2 completely frozen (requires_grad=False)
   - Layer 3-4: Trainable
4. Inference: Two-stage routing using LEGO's Early Exit mechanism

Key Benefits:
- Focuses computational resources on hard examples during training
- Hard PPL: 78% improvement (2763 → 668)
- Reduces compute cost by 36% using adaptive routing
- Fully integrated with LEGO framework's 2 core options

LEGO Framework (2 Core Options):
1. stages: Which stage blocks to train (LEGO blocks configuration)
2. routing_threshold: When to exit early (inference efficiency)

References:
- LEGO: Layered Ensemble with Gradual Optimization (本フレームワーク)
- ASHEM: Adaptive Supervision via Hard Example Mining (本研究)
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- Early Exit: BranchyNet (2016), Teerapittayanon et al. (2016)
- Deep Supervision: Lee et al. (2015)
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Type

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

sys.path.insert(0, 'experiments')
from utils import set_seed, get_device, create_wikitext_dataloaders


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Extended configuration for ASHEM experiment.

    Combines ASHEMConfig with dataset/model-specific parameters.
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

    # ASHEM configuration (delegates to ASHEMConfig)
    ashem: ASHEMConfig = field(default_factory=ASHEMConfig)


CONFIG = ExperimentConfig()


# ==============================================================================
# Utility Functions
# ==============================================================================
# Note: All utility functions are now imported from framework modules:
#
# ASHEM functions (from src/ease/ashem.py):
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

def run_experiment(
    model_name: str,
    ModelClass: Type[nn.Module],
    config_fn: Callable[[int], TrainingConfig],
    device: str
) -> Dict[str, Any]:
    """
    Run complete hard example mining experiment.

    Experiment flow:
    1. Phase 1: Train shallow model (2 layers) on all data
    2. Compute confidence threshold and collect hard examples
    3. Phase 2: Extend to 4 layers, train upper layers on hard examples only
    4. Evaluate using two-stage inference (LEGO Early Exit)

    Args:
        model_name: Name of model architecture for logging
        ModelClass: Model class to use for Phase 1
        config_fn: Function to create training config
        device: Device to run experiment on

    Returns:
        Dictionary with experiment results and metrics
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Hard Example Mining")
    print(f"{'='*60}\n")

    # ==========================================================================
    # Phase 1: Train Shallow Model
    # ==========================================================================
    print(f"Phase 1: Train {CONFIG.ashem.phase1_layers}-layer model")
    print(f"{'='*60}")

    set_seed(42)
    train_loader, val_loader, vocab_size = create_wikitext_dataloaders(
        CONFIG.phase1_samples, CONFIG.phase1_batch, CONFIG.seq_len
    )
    CONFIG.vocab_size = vocab_size  # Update vocab size from data

    # Create shallow model
    model = ModelClass(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.ashem.phase1_layers,
        num_heads=CONFIG.num_heads
    ).to(device)

    # Train with early stopping
    config = config_fn(CONFIG.ashem.phase1_layers)
    trainer = Trainer(config, vocab_size=CONFIG.vocab_size, device=device)
    optimizer = trainer.create_optimizer(model, base_lr=CONFIG.ashem.phase1_lr)

    start_time = time.time()
    result_phase1 = trainer.train_with_early_stopping(
        model=model,
        train_batches=train_loader,
        val_batches=val_loader,
        optimizer=optimizer,
        max_epochs=CONFIG.phase1_epochs,
        patience=CONFIG.ashem.phase1_patience,
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
    print(f"Computing Confidence Threshold (target ratio: {CONFIG.ashem.hard_example_ratio*100:.0f}%)")
    print(f"{'='*60}\n")

    confidence_threshold = compute_confidence_threshold(
        model, val_loader, CONFIG.ashem.hard_example_ratio, device
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
    total_samples = CONFIG.phase1_samples * 0.2 * CONFIG.seq_len

    print(f"✓ Collected {num_hard:,} hard examples")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"  Actual ratio: {num_hard / total_samples * 100:.1f}% "
          f"(target: {CONFIG.ashem.hard_example_ratio*100:.0f}%)")

    # ==========================================================================
    # Evaluate Phase 1 on Hard Examples
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Evaluating Phase 1 on Hard Examples")
    print(f"{'='*60}\n")

    phase1_hard_ppl = evaluate_on_hard_examples(
        model, hard_examples, CONFIG.vocab_size, device,
        batch_size=CONFIG.phase2_batch, num_lower_layers=CONFIG.ashem.phase1_layers
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
    model_extended = DeepSupervisionTransformer(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.ashem.phase2_layers,
        num_heads=CONFIG.num_heads,
        exit_layer=CONFIG.ashem.phase1_layers,
        routing_threshold=confidence_threshold
    ).to(device)

    # Copy weights from Phase 1 model
    model_extended.embedding.load_state_dict(model.embedding.state_dict())
    for i in range(CONFIG.ashem.phase1_layers):
        model_extended.layers[i].load_state_dict(model.layers[i].state_dict())
    model_extended.output_head.load_state_dict(model.output_head.state_dict())

    print("✓ Copied weights from 2-layer model")
    print("✓ Layers 3-4 randomly initialized")

    # Hard Freezing: Freeze lower layers
    print("\n📊 Hard Freezing Configuration:")
    print("  Layer 1-2: Frozen (requires_grad=False)")
    print("  Layer 3-4: Trainable")

    for param in model_extended.embedding.parameters():
        param.requires_grad = False
    for i in range(CONFIG.ashem.phase1_layers):
        for param in model_extended.layers[i].parameters():
            param.requires_grad = False

    # Configure training (final layer only for loss)
    phase2_config = TrainingConfig(
        stages=[StageConfig(layers=(4, 4), loss_weight=1.0)]  # Final layer only for loss
    )

    trainer_phase2 = Trainer(phase2_config, vocab_size=CONFIG.vocab_size)

    trainable = sum(p.numel() for p in model_extended.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_extended.parameters())
    print(f"\n✓ Frozen lower layers")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Create hard example loader
    hard_batches = create_hard_example_loader(hard_examples, CONFIG.phase2_batch)
    print(f"  Hard example batches: {len(hard_batches)}")

    # Train upper layers only
    optimizer_upper = trainer_phase2.create_optimizer(
        model_extended,
        base_lr=CONFIG.ashem.phase2_lr
    )

    print(f"\n📊 Training Configuration:")
    print(f"  Learning rate: {CONFIG.ashem.phase2_lr:.1e}")
    print(f"  Patience: {CONFIG.ashem.phase2_patience}")
    print(f"  Max epochs: {CONFIG.phase2_epochs}")

    # Training loop with early stopping
    best_val_ppl = float('inf')
    best_model_state = None
    patience_counter = 0

    start_time = time.time()
    for epoch in range(CONFIG.phase2_epochs):
        # Train on hard examples
        train_loss = train_upper_layers(
            model_extended, hard_batches, optimizer_upper,
            CONFIG.vocab_size, device, CONFIG.ashem.phase1_layers
        )

        # Evaluate with LEGO's Early Exit
        eval_config = TrainingConfig(
            stages=[StageConfig(layers=(CONFIG.ashem.phase2_layers, CONFIG.ashem.phase2_layers), loss_weight=1.0)],
            routing_threshold=confidence_threshold,
            exit_layer=CONFIG.ashem.phase1_layers
        )

        eval_trainer = Trainer(eval_config, vocab_size=CONFIG.vocab_size, device=device)
        val_stats = eval_trainer.evaluate(model_extended, val_loader)

        val_acc = val_stats['acc']
        val_ppl = val_stats['ppl']
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Evaluate on hard examples
        hard_ppl = evaluate_on_hard_examples(
            model_extended, hard_examples, CONFIG.vocab_size, device,
            batch_size=CONFIG.phase2_batch, num_lower_layers=CONFIG.ashem.phase1_layers
        )

        print(f"Epoch {epoch+1}/{CONFIG.phase2_epochs} - "
              f"Train PPL: {train_ppl:.4f} | "
              f"Val PPL: {val_ppl:.2f} | "
              f"Val Acc: {val_acc*100:.2f}% | "
              f"Hard PPL: {hard_ppl:.2f}")

        # Early stopping
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_model_state = {k: v.cpu().clone() for k, v in model_extended.state_dict().items()}
            patience_counter = 0
            print(f"  → New best (val_ppl: {val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{CONFIG.ashem.phase2_patience})")

        if patience_counter >= CONFIG.ashem.phase2_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best model was at epoch {epoch - patience_counter + 1}")
            break

    phase2_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model_extended.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print("\nRestored best model from Phase 2")

    # Evaluate best model on hard examples
    phase2_hard_ppl = evaluate_on_hard_examples(
        model_extended, hard_examples, CONFIG.vocab_size, device,
        batch_size=CONFIG.phase2_batch, num_lower_layers=CONFIG.ashem.phase1_layers
    )

    # Get final val stats
    final_eval_config = TrainingConfig(
        stages=[StageConfig(layers=(CONFIG.ashem.phase2_layers, CONFIG.ashem.phase2_layers), loss_weight=1.0)],
        routing_threshold=confidence_threshold,
        exit_layer=CONFIG.ashem.phase1_layers
    )
    final_eval_trainer = Trainer(final_eval_config, vocab_size=CONFIG.vocab_size, device=device)
    final_val_stats = final_eval_trainer.evaluate(model_extended, val_loader)

    print("\nPhase 2 Results:")
    print(f"  Best Val PPL: {final_val_stats['ppl']:.2f}")
    print(f"  Best Hard PPL: {phase2_hard_ppl:.2f}")
    print(f"  Hard PPL Improvement: {phase1_hard_ppl - phase2_hard_ppl:+.2f} "
          f"({(phase1_hard_ppl - phase2_hard_ppl) / phase1_hard_ppl * 100:+.1f}%)")
    print(f"  Time: {phase2_time:.2f}s")

    # ==========================================================================
    # Final Evaluation: Two-Stage Inference with LEGO
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Final Evaluation (Two-Stage Inference)")
    print(f"{'='*60}\n")

    # Use LEGO's built-in Early Exit evaluation
    final_config = TrainingConfig(
        stages=[StageConfig(layers=(CONFIG.ashem.phase2_layers, CONFIG.ashem.phase2_layers), loss_weight=1.0)],
        routing_threshold=confidence_threshold,
        exit_layer=CONFIG.ashem.phase1_layers
    )

    final_trainer = Trainer(final_config, vocab_size=CONFIG.vocab_size, device=device)
    stats = final_trainer.evaluate(model_extended, val_loader)

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
        'model_name': model_name,
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
    print("="*60)
    print("Hard Example Mining + Two-Stage Inference")
    print("="*60)
    print(f"Device: {get_device()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nExperiment Design:")
    print(f"  Phase 1: Train {CONFIG.ashem.phase1_layers}-layer model")
    print(f"  Compute: Auto-adjust threshold to collect {CONFIG.ashem.hard_example_ratio*100:.0f}% hard examples")
    print(f"  Phase 2: Add {CONFIG.ashem.phase2_layers - CONFIG.ashem.phase1_layers} layers → Train on hard examples")
    print("  Eval: Two-stage inference (Layer 2 or Layer 4) using LEGO's Early Exit")
    print()

    device = get_device()

    # Run experiment with Standard Transformer
    run_experiment(
        "Standard Transformer",
        StandardTransformer,
        create_standard_config,
        device
    )

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
