"""
LEGO (Layered Ensemble with Gradual Optimization) Experiment

**Dataset**: WikiText-2 (10K samples)

Experiment flow:
1. Phase 1: Train shallow layers on all data
2. Collect hard examples using confidence threshold
3. Phase 2: Train deeper layers on hard examples only
4. Inference: Two-stage routing using learned thresholds
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
from dataclasses import dataclass
from typing import Dict, Any

from ease import (
    LEGOTransformer,
    Trainer,
    TrainingConfig,
    StageConfig,
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
    """Configuration for LEGO experiment."""
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

def run_experiment(device: str) -> Dict[str, Any]:
    """Run LEGO experiment."""
    print(f"\n{'='*60}")
    print("LEGO Cascading Training")
    print(f"{'='*60}\n")

    # Setup Data
    set_seed(42)
    train_loader, val_loader, vocab_size = create_wikitext_dataloaders(
        CONFIG.num_samples, CONFIG.batch_size, CONFIG.seq_len
    )
    CONFIG.vocab_size = vocab_size

    # Create Model
    model = LEGOTransformer(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.phase2_layers,
        num_heads=CONFIG.num_heads,
        exit_layer=CONFIG.phase1_layers,
        routing_threshold=0.5
    ).to(device)

    # LEGO Config
    lego_config = LEGOConfig(
        phases=[
            PhaseConfig(
                layers=(1, CONFIG.phase1_layers),
                lr=CONFIG.phase1_lr,
                patience=CONFIG.phase1_patience,
                max_epochs=CONFIG.phase1_max_epochs,
                freeze_lower=False
            ),
            PhaseConfig(
                layers=(CONFIG.phase1_layers + 1, CONFIG.phase2_layers),
                lr=CONFIG.phase2_lr,
                patience=CONFIG.phase2_patience,
                max_epochs=CONFIG.phase2_max_epochs,
                freeze_lower=True
            ),
        ],
        hard_example_ratio=CONFIG.hard_example_ratio,
    )

    print(f"Config: {lego_config.describe()}")
    print(f"Model: {CONFIG.phase2_layers} layers ({CONFIG.dim} dim)")

    # Train
    trainer = LEGOTrainer(lego_config, vocab_size=CONFIG.vocab_size, device=device, verbose=True)

    start_time = time.time()
    result = trainer.train(model, train_loader, val_loader, batch_size=CONFIG.batch_size)
    total_time = time.time() - start_time

    thresholds = result['thresholds']
    phase_histories = result['phase_histories']
    hard_examples = result['hard_examples']

    # Evaluate Hard Examples
    print(f"\n{'='*60}")
    print("Hard Example Performance")
    print(f"{'='*60}\n")

    if hard_examples is not None:
        phase2_hard_ppl = evaluate_on_hard_examples(
            model, hard_examples, CONFIG.vocab_size, device,
            batch_size=CONFIG.batch_size, num_lower_layers=CONFIG.phase1_layers
        )
        print(f"Phase 2 Hard PPL: {phase2_hard_ppl:.2f}")

        num_hard = len(hard_examples['targets'])
        total_tokens = sum(x.numel() for x, _ in val_loader)
        print(f"Hard tokens: {num_hard}/{total_tokens} ({num_hard/total_tokens*100:.1f}%)")

    # Final Evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation (Two-Stage Inference)")
    print(f"{'='*60}\n")

    if thresholds:
        model.routing_threshold = thresholds[0]

    final_config = TrainingConfig(
        stages=[StageConfig(layers=(CONFIG.phase2_layers, CONFIG.phase2_layers), loss_weight=1.0)],
        routing_threshold=model.routing_threshold,
        exit_layer=CONFIG.phase1_layers
    )

    final_trainer = Trainer(final_config, vocab_size=CONFIG.vocab_size, device=device)
    stats = final_trainer.evaluate(model, val_loader)

    print(f"Accuracy: {stats['acc']*100:.2f}%")
    print(f"PPL: {stats['ppl']:.2f}")
    print(f"Shallow ratio: {stats['shallow_ratio']*100:.1f}%")
    print(f"Compute cost: {stats['compute_cost']:.2%}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Phase 1 epochs: {phase_histories[0]['total_epochs']}")
    print(f"Phase 2 epochs: {phase_histories[1]['total_epochs']}")
    print(f"Threshold: {thresholds[0]:.4f}" if thresholds else "No threshold")
    print(f"Total time: {total_time:.2f}s")

    return {
        'thresholds': thresholds,
        'final_acc': stats['acc'] * 100,
        'final_ppl': stats['ppl'],
        'shallow_ratio': stats['shallow_ratio'],
        'compute_cost': stats['compute_cost'],
        'total_time': total_time,
    }


def main() -> None:
    """Run experiment."""
    print("="*60)
    print("LEGO: Layered Ensemble with Gradual Optimization")
    print("="*60)
    print(f"Device: {get_device()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nPhase 1: {CONFIG.phase1_layers} layers on all data")
    print(f"Phase 2: +{CONFIG.phase2_layers - CONFIG.phase1_layers} layers on hard examples")

    run_experiment(get_device())

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
