"""
LEGO Framework - ASHEM Training Strategy Demo

Pure LEGO design implementation of ASHEM (Adaptive Supervision via Hard Example Mining).

LEGO's 2 Core Options:
1. StageConfig(layers=(x, y), loss_weight=w) - Which layers compute loss
2. routing_threshold + exit_layer - Early exit at inference

ASHEM Strategy (2-Phase Training):
- Phase 1: Train Block 1 (layers 1-2) on all data
- Phase 2: Extend to Block 2 (layers 3-4), train only on hard examples
- Inference: Two-stage routing using LEGO Early Exit

Key Design Principles:
- Use standard Trainer workflow for both phases
- Hard examples = full sequences (not individual tokens)
- Layer freezing via freeze_lower_layers()
- All training through Trainer.train_with_early_stopping()
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
from dataclasses import dataclass, field

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
    freeze_lower_layers,
    get_trainable_params_info,
)

sys.path.insert(0, 'experiments')
from utils import set_seed, get_device, create_wikitext_dataloaders


@dataclass
class ExperimentConfig:
    """Experiment configuration combining model and ASHEM parameters."""
    # Model
    vocab_size: int = 69830
    seq_len: int = 32
    dim: int = 64
    num_heads: int = 4

    # Training
    num_samples: int = 10000
    batch_size: int = 64
    max_epochs: int = 50

    # ASHEM
    ashem: ASHEMConfig = field(default_factory=ASHEMConfig)


CONFIG = ExperimentConfig()


def main() -> None:
    """Run ASHEM experiment using pure LEGO design."""
    print("=" * 60)
    print("LEGO ASHEM: Pure Design Implementation")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    print("\nLEGO Design:")
    print("  Phase 1: StageConfig(layers=(2, 2)) - Train Block 1")
    print("  Phase 2: StageConfig(layers=(4, 4)) - Train Block 2 on hard examples")
    print("  Inference: routing_threshold + exit_layer=2")
    print()

    # ==========================================================================
    # Data Preparation
    # ==========================================================================
    set_seed(42)
    train_loader, val_loader, vocab_size = create_wikitext_dataloaders(
        CONFIG.num_samples, CONFIG.batch_size, CONFIG.seq_len
    )
    CONFIG.vocab_size = vocab_size

    # ==========================================================================
    # Phase 1: Train Block 1 (Layers 1-2)
    # ==========================================================================
    print("=" * 60)
    print("Phase 1: Train Block 1 (2 layers)")
    print("=" * 60)

    # Create shallow model
    model = StandardTransformer(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.ashem.phase1_layers,
        num_heads=CONFIG.num_heads
    ).to(device)

    # LEGO config: loss at final layer only
    config = create_standard_config(CONFIG.ashem.phase1_layers)
    trainer = Trainer(config, vocab_size=CONFIG.vocab_size, device=device)
    optimizer = trainer.create_optimizer(model, base_lr=CONFIG.ashem.phase1_lr)

    print(f"\nConfig: {config.describe()}")

    start_time = time.time()
    result = trainer.train_with_early_stopping(
        model=model,
        train_batches=train_loader,
        val_batches=val_loader,
        optimizer=optimizer,
        max_epochs=CONFIG.max_epochs,
        patience=CONFIG.ashem.phase1_patience,
        verbose=True
    )
    phase1_time = time.time() - start_time

    phase1_stats = trainer.evaluate(model, val_loader)
    print(f"\nPhase 1 Results:")
    print(f"  Accuracy: {phase1_stats['acc']*100:.2f}%")
    print(f"  PPL: {phase1_stats['ppl']:.2f}")
    print(f"  Time: {phase1_time:.1f}s")

    # ==========================================================================
    # Compute Confidence Threshold
    # ==========================================================================
    print("\n" + "=" * 60)
    print(f"Computing Threshold (target: {CONFIG.ashem.hard_example_ratio*100:.0f}% hard)")
    print("=" * 60)

    threshold = compute_confidence_threshold(
        model, val_loader, CONFIG.ashem.hard_example_ratio, device
    )
    print(f"\nThreshold: {threshold:.4f}")

    # ==========================================================================
    # Collect Hard Examples
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Collecting Hard Examples")
    print("=" * 60)

    # Collect from training data (for Phase 2 training)
    hard_train = collect_hard_examples(model, train_loader, threshold, device)
    hard_val = collect_hard_examples(model, val_loader, threshold, device)

    total_train_seqs = sum(x.shape[0] for x, _ in train_loader)
    hard_train_seqs = sum(x.shape[0] for x, _ in hard_train)
    print(f"\nTrain: {hard_train_seqs}/{total_train_seqs} sequences ({hard_train_seqs/total_train_seqs*100:.1f}%)")
    print(f"Val: {sum(x.shape[0] for x, _ in hard_val)} hard sequences")

    # ==========================================================================
    # Phase 2: Extend to Block 2, Train on Hard Examples
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Add Block 2 (layers 3-4), train on hard examples")
    print("=" * 60)

    # Create extended model with Early Exit support
    model_extended = DeepSupervisionTransformer(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.ashem.phase2_layers,
        num_heads=CONFIG.num_heads,
        exit_layer=CONFIG.ashem.phase1_layers,
        routing_threshold=threshold
    ).to(device)

    # Copy weights from Phase 1
    model_extended.embedding.load_state_dict(model.embedding.state_dict())
    for i in range(CONFIG.ashem.phase1_layers):
        model_extended.layers[i].load_state_dict(model.layers[i].state_dict())
    model_extended.output_head.load_state_dict(model.output_head.state_dict())

    print("\nWeight transfer: Block 1 copied from Phase 1")

    # Freeze lower layers (LEGO Hard Freezing)
    freeze_lower_layers(model_extended, CONFIG.ashem.phase1_layers)
    params_info = get_trainable_params_info(model_extended)
    print(f"Hard Freezing: {params_info['trainable']:,}/{params_info['total']:,} params trainable ({params_info['ratio']*100:.1f}%)")

    # LEGO config: loss at final layer (layer 4)
    phase2_config = TrainingConfig(
        stages=[StageConfig(layers=(CONFIG.ashem.phase2_layers, CONFIG.ashem.phase2_layers), loss_weight=1.0)]
    )
    trainer2 = Trainer(phase2_config, vocab_size=CONFIG.vocab_size, device=device)

    # Only train parameters with requires_grad=True
    optimizer2 = torch.optim.AdamW(
        [p for p in model_extended.parameters() if p.requires_grad],
        lr=CONFIG.ashem.phase2_lr
    )

    print(f"\nConfig: {phase2_config.describe()}")

    start_time = time.time()
    result2 = trainer2.train_with_early_stopping(
        model=model_extended,
        train_batches=hard_train,  # Train only on hard examples
        val_batches=hard_val,      # Validate on hard examples
        optimizer=optimizer2,
        max_epochs=CONFIG.max_epochs,
        patience=CONFIG.ashem.phase2_patience,
        verbose=True
    )
    phase2_time = time.time() - start_time

    print(f"\nPhase 2 Time: {phase2_time:.1f}s")

    # ==========================================================================
    # Final Evaluation: Two-Stage Inference
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Final Evaluation: Two-Stage Inference (LEGO Early Exit)")
    print("=" * 60)

    # Enable routing for inference
    eval_config = TrainingConfig(
        stages=[StageConfig(layers=(CONFIG.ashem.phase2_layers, CONFIG.ashem.phase2_layers), loss_weight=1.0)],
        routing_threshold=threshold,
        exit_layer=CONFIG.ashem.phase1_layers
    )
    eval_trainer = Trainer(eval_config, vocab_size=CONFIG.vocab_size, device=device)
    final_stats = eval_trainer.evaluate(model_extended, val_loader)

    print(f"\nTwo-Stage Results:")
    print(f"  Accuracy: {final_stats['acc']*100:.2f}%")
    print(f"  PPL: {final_stats['ppl']:.2f}")
    print(f"  Shallow ratio: {final_stats['shallow_ratio']*100:.1f}%")
    print(f"  Compute cost: {final_stats['compute_cost']*100:.1f}%")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nPhase 1 (Block 1 only):")
    print(f"  Acc: {phase1_stats['acc']*100:.2f}% | PPL: {phase1_stats['ppl']:.2f}")
    print("\nPhase 2 (Two-Stage):")
    print(f"  Acc: {final_stats['acc']*100:.2f}% | PPL: {final_stats['ppl']:.2f}")
    print(f"\nEfficiency:")
    print(f"  Shallow exits: {final_stats['shallow_ratio']*100:.1f}%")
    print(f"  Compute saved: {(1-final_stats['compute_cost'])*100:.1f}%")

    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
