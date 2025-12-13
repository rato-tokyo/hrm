"""
LEGO (Layered Ensemble with Gradual Optimization) Experiment

**Experiment Version**: LEGO with Cascading Hard Example Mining
**Date**: 2025-12-13
**Dataset**: WikiText-2 (10K samples)

This experiment demonstrates the LEGO training strategy:
1. Phase 1: Train a shallow model (2 layers) on all data
2. Identify hard examples using confidence-based threshold (auto-adjusted)
3. Phase 2: Train additional layers (2 more) on hard examples only
   - Hard Freezing: Layer 1-2 completely frozen (requires_grad=False)
   - Layer 3-4: Trainable
4. Inference: Two-stage routing using LEGO's Early Exit mechanism

Key Benefits:
- Focuses computational resources on hard examples during training
- Hard PPL: ~20% improvement through cascading training
- Reduces compute cost using adaptive routing
- Flexible multi-phase configuration

LEGO Framework Core Concepts:
- PhaseConfig: Configuration for each training phase (layers, lr, patience)
- LEGOConfig: Multi-phase cascading training configuration
- LEGOTrainer: Automated cascading training with hard example collection

References:
- LEGO: Layered Ensemble with Gradual Optimization (本フレームワーク)
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- Early Exit: BranchyNet (2016), Teerapittayanon et al. (2016)
- Deep Supervision: Lee et al. (2015)
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Callable, Type

from ease import (
    StandardTransformer,
    DeepSupervisionTransformer,
    Trainer,
    TrainingConfig,
    StageConfig,
    create_standard_config,
    LEGOConfig,
    PhaseConfig,
    LEGOTrainer,
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

    Combines LEGOConfig with dataset/model-specific parameters.
    """
    # Model architecture
    vocab_size: int = 69830
    seq_len: int = 32
    dim: int = 64
    num_heads: int = 4

    # Dataset parameters
    num_samples: int = 10000
    batch_size: int = 64

    # LEGO configuration
    phase1_layers: int = 2
    phase1_lr: float = 1e-3
    phase1_patience: int = 1
    phase1_max_epochs: int = 50

    phase2_layers: int = 4
    phase2_lr: float = 1e-4
    phase2_patience: int = 3
    phase2_max_epochs: int = 50

    hard_example_ratio: float = 0.5


CONFIG = ExperimentConfig()


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
    Run complete LEGO experiment with cascading hard example mining.

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
    print(f"{model_name} - LEGO Cascading Training")
    print(f"{'='*60}\n")

    # ==========================================================================
    # Setup Data
    # ==========================================================================
    set_seed(42)
    train_loader, val_loader, vocab_size = create_wikitext_dataloaders(
        CONFIG.num_samples, CONFIG.batch_size, CONFIG.seq_len
    )
    CONFIG.vocab_size = vocab_size

    # ==========================================================================
    # Create Model and LEGO Config
    # ==========================================================================
    model = DeepSupervisionTransformer(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.phase2_layers,  # Create full 4-layer model
        num_heads=CONFIG.num_heads,
        exit_layer=CONFIG.phase1_layers,
        routing_threshold=0.5  # Will be updated after Phase 1
    ).to(device)

    lego_config = LEGOConfig(
        phases=[
            PhaseConfig(
                layers=(1, CONFIG.phase1_layers),
                lr=CONFIG.phase1_lr,
                patience=CONFIG.phase1_patience,
                max_epochs=CONFIG.phase1_max_epochs,
                freeze_lower=False  # Phase 1 trains from scratch
            ),
            PhaseConfig(
                layers=(CONFIG.phase1_layers + 1, CONFIG.phase2_layers),
                lr=CONFIG.phase2_lr,
                patience=CONFIG.phase2_patience,
                max_epochs=CONFIG.phase2_max_epochs,
                freeze_lower=True  # Phase 2 freezes lower layers
            ),
        ],
        hard_example_ratio=CONFIG.hard_example_ratio,
    )

    print(f"LEGO Config: {lego_config.describe()}")
    print(f"Model: {CONFIG.phase2_layers} layers ({CONFIG.dim} dim, {CONFIG.num_heads} heads)")
    print(f"Dataset: {CONFIG.num_samples} samples, seq_len={CONFIG.seq_len}")

    # ==========================================================================
    # Train with LEGOTrainer
    # ==========================================================================
    trainer = LEGOTrainer(
        lego_config,
        vocab_size=CONFIG.vocab_size,
        device=device,
        verbose=True
    )

    start_time = time.time()
    result = trainer.train(model, train_loader, val_loader, batch_size=CONFIG.batch_size)
    total_time = time.time() - start_time

    # Extract results
    thresholds = result['thresholds']
    phase_histories = result['phase_histories']
    hard_examples = result['hard_examples']

    # ==========================================================================
    # Evaluate Phase 1 and Phase 2 Hard PPL
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Hard Example Performance")
    print(f"{'='*60}\n")

    if hard_examples is not None:
        # Phase 2 Hard PPL (after training)
        phase2_hard_ppl = evaluate_on_hard_examples(
            model, hard_examples, CONFIG.vocab_size, device,
            batch_size=CONFIG.batch_size, num_lower_layers=CONFIG.phase1_layers
        )
        print(f"Phase 2 Hard PPL: {phase2_hard_ppl:.2f}")

        num_hard = len(hard_examples['targets'])
        total_tokens = sum(x.numel() for x, _ in val_loader)
        hard_ratio = num_hard / total_tokens
        print(f"Hard tokens: {num_hard}/{total_tokens} ({hard_ratio*100:.1f}%)")

    # ==========================================================================
    # Final Evaluation with Two-Stage Inference
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Final Evaluation (Two-Stage Inference)")
    print(f"{'='*60}\n")

    # Update model threshold
    if thresholds:
        model.routing_threshold = thresholds[0]

    final_config = TrainingConfig(
        stages=[StageConfig(layers=(CONFIG.phase2_layers, CONFIG.phase2_layers), loss_weight=1.0)],
        routing_threshold=model.routing_threshold,
        exit_layer=CONFIG.phase1_layers
    )

    final_trainer = Trainer(final_config, vocab_size=CONFIG.vocab_size, device=device)
    stats = final_trainer.evaluate(model, val_loader)

    print("Results:")
    print(f"  Accuracy: {stats['acc']*100:.2f}%")
    print(f"  PPL: {stats['ppl']:.2f}")
    print(f"  Shallow ratio (Layer {CONFIG.phase1_layers}): {stats['shallow_ratio']*100:.1f}%")
    print(f"  Deep ratio (Layer {CONFIG.phase2_layers}): {(1-stats['shallow_ratio'])*100:.1f}%")
    print(f"  Compute cost: {stats['compute_cost']:.2%} of full model")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"\nPhase 1 epochs: {phase_histories[0]['total_epochs']}")
    print(f"Phase 2 epochs: {phase_histories[1]['total_epochs']}")
    print(f"Threshold: {thresholds[0]:.4f}" if thresholds else "No threshold")
    print(f"Total time: {total_time:.2f}s")

    return {
        'model_name': model_name,
        'thresholds': thresholds,
        'phase1_epochs': phase_histories[0]['total_epochs'],
        'phase2_epochs': phase_histories[1]['total_epochs'],
        'final_acc': stats['acc'] * 100,
        'final_ppl': stats['ppl'],
        'shallow_ratio': stats['shallow_ratio'],
        'compute_cost': stats['compute_cost'],
        'total_time': total_time,
    }


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main() -> None:
    """Run LEGO experiment."""
    print("="*60)
    print("LEGO: Layered Ensemble with Gradual Optimization")
    print("="*60)
    print(f"Device: {get_device()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nExperiment Design:")
    print(f"  Phase 1: Train {CONFIG.phase1_layers}-layer model on all data")
    print(f"  Collect: {CONFIG.hard_example_ratio*100:.0f}% hard examples")
    print(f"  Phase 2: Add {CONFIG.phase2_layers - CONFIG.phase1_layers} layers, train on hard examples")
    print("  Eval: Two-stage inference (LEGO Early Exit)")
    print()

    device = get_device()

    # Run experiment
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
