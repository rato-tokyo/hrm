"""
Progressive Layer Training Experiment

Experiment: Train 3 layers → Freeze them → Add 1 layer → Train only the new layer

This tests if Deep Supervision enables better progressive training.
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
import torch.nn as nn
from dataclasses import dataclass

from ease import (
    StandardTransformer,
    Trainer,
    create_standard_config,
)

# データローダー
from colab import create_dataloaders as create_dataloaders_from_colab


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get device (cuda or cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class Config:
    """Experiment configuration"""
    vocab_size: int = 69830  # Will be set from WikiText-2
    seq_len: int = 32
    dim: int = 64
    initial_layers: int = 3  # Initial number of layers
    added_layers: int = 1    # Layers to add
    num_heads: int = 4
    base_lr: float = 1e-3

    # Phase 1: Initial training (3 layers)
    phase1_samples: int = 10000
    phase1_batch: int = 64
    phase1_epochs: int = 50
    phase1_patience: int = 1

    # Phase 2: Progressive training (freeze 3 layers, train new 1 layer)
    phase2_samples: int = 10000
    phase2_batch: int = 64
    phase2_epochs: int = 50
    phase2_patience: int = 1

CONFIG = Config()


def create_dataloaders(num_samples: int, batch_size: int, seq_len: int):
    """Create train and validation dataloaders."""
    train_loader, val_loader = create_dataloaders_from_colab(num_samples, batch_size, seq_len)
    return train_loader, val_loader


def add_layer_to_model(model: nn.Module, device: str) -> nn.Module:
    """
    Add one layer to existing model.

    Creates a new model with N+1 layers, copying weights from N-layer model.
    """
    if isinstance(model, StandardTransformer):
        new_model = StandardTransformer(
            vocab_size=model.vocab_size,
            dim=model.dim,
            num_layers=model.num_layers + 1,
            num_heads=4
        ).to(device)
    else:
        new_model = DeepSupervisionTransformer(
            vocab_size=model.vocab_size,
            dim=model.dim,
            num_layers=model.num_layers + 1,
            num_heads=4
        ).to(device)

    # Copy embedding and output_head
    new_model.embedding.load_state_dict(model.embedding.state_dict())
    new_model.output_head.load_state_dict(model.output_head.state_dict())

    # Copy existing layers
    for i in range(model.num_layers):
        new_model.layers[i].load_state_dict(model.layers[i].state_dict())

    # New layer (layers[3]) is randomly initialized
    print(f"✓ Added new layer {model.num_layers + 1} (randomly initialized)")

    return new_model


def freeze_initial_layers(model: nn.Module, num_layers_to_freeze: int):
    """Freeze first N layers."""
    # Freeze embedding
    for param in model.embedding.parameters():
        param.requires_grad = False

    # Freeze initial layers
    for i in range(num_layers_to_freeze):
        for param in model.layers[i].parameters():
            param.requires_grad = False

    # Keep output_head trainable
    for param in model.output_head.parameters():
        param.requires_grad = True

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"✓ Frozen first {num_layers_to_freeze} layers")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def run_progressive_experiment(model_name: str, ModelClass, config_fn, device: str):
    """
    Run progressive layer training experiment.

    Steps:
    1. Train 3-layer model normally
    2. Add 1 layer → Freeze first 3 layers → Train only new layer
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Progressive Layer Training")
    print(f"{'='*60}\n")

    # ========================================
    # Phase 1: Train initial 3-layer model
    # ========================================
    print(f"Phase 1: Train {CONFIG.initial_layers}-layer model")
    print(f"{'='*60}")

    set_seed(42)
    train_loader, val_loader = create_dataloaders(
        CONFIG.phase1_samples, CONFIG.phase1_batch, CONFIG.seq_len
    )

    # Create 3-layer model
    model = ModelClass(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.initial_layers,
        num_heads=CONFIG.num_heads
    ).to(device)

    config = config_fn(num_layers=CONFIG.initial_layers)
    trainer = Trainer(config, vocab_size=CONFIG.vocab_size, device=device)
    optimizer = trainer.create_optimizer(model, base_lr=CONFIG.base_lr)

    start_time = time.time()
    result_phase1 = trainer.train_with_early_stopping(
        model=model,
        train_batches=train_loader,
        val_batches=val_loader,
        optimizer=optimizer,
        max_epochs=CONFIG.phase1_epochs,
        patience=CONFIG.phase1_patience,
        verbose=True
    )
    phase1_time = time.time() - start_time

    phase1_acc = result_phase1['val_accs'][result_phase1['best_epoch']] * 100
    phase1_ppl = result_phase1['val_losses'][result_phase1['best_epoch']]

    print("\nPhase 1 Results:")
    print(f"  Best Acc: {phase1_acc:.2f}%")
    print(f"  Best PPL: {phase1_ppl:.2f}")
    print(f"  Time: {phase1_time:.2f}s")

    # ========================================
    # Phase 2: Add layer + Freeze + Train
    # ========================================
    print(f"\n{'='*60}")
    print("Phase 2: Add 1 layer → Freeze first 3 → Train new layer")
    print(f"{'='*60}\n")

    # Add new layer
    model_extended = add_layer_to_model(model, device)

    # Freeze first 3 layers
    freeze_initial_layers(model_extended, num_layers_to_freeze=CONFIG.initial_layers)

    # Create new config for 4-layer model
    config_extended = config_fn(num_layers=CONFIG.initial_layers + CONFIG.added_layers)
    trainer_extended = Trainer(config_extended, vocab_size=CONFIG.vocab_size, device=device)

    # Only train new layer + output_head with reduced learning rate
    # Lower LR for new layer to avoid overshooting
    reduced_lr = CONFIG.base_lr * 0.1  # 1e-3 → 1e-4
    optimizer_extended = trainer_extended.create_optimizer(model_extended, base_lr=reduced_lr)

    print(f"  Using reduced learning rate: {reduced_lr:.1e} (base_lr × 0.1)")

    # Train
    train_loader2, val_loader2 = create_dataloaders(
        CONFIG.phase2_samples, CONFIG.phase2_batch, CONFIG.seq_len
    )

    start_time = time.time()
    result_phase2 = trainer_extended.train_with_early_stopping(
        model=model_extended,
        train_batches=train_loader2,
        val_batches=val_loader2,
        optimizer=optimizer_extended,
        max_epochs=CONFIG.phase2_epochs,
        patience=CONFIG.phase2_patience,
        verbose=True
    )
    phase2_time = time.time() - start_time

    phase2_acc = result_phase2['val_accs'][result_phase2['best_epoch']] * 100
    phase2_ppl = result_phase2['val_losses'][result_phase2['best_epoch']]

    print("\nPhase 2 Results:")
    print(f"  Best Acc: {phase2_acc:.2f}%")
    print(f"  Best PPL: {phase2_ppl:.2f}")
    print(f"  Time: {phase2_time:.2f}s")

    # Total
    total_time = phase1_time + phase2_time
    acc_improvement = phase2_acc - phase1_acc
    ppl_improvement = ((phase1_ppl - phase2_ppl) / phase1_ppl) * 100

    print(f"\n{'='*60}")
    print(f"Summary: {model_name}")
    print(f"{'='*60}")
    print(f"Phase 1 (3 layers):  Acc {phase1_acc:.2f}% | PPL {phase1_ppl:.2f} | {phase1_time:.2f}s")
    print(f"Phase 2 (+1 layer):  Acc {phase2_acc:.2f}% | PPL {phase2_ppl:.2f} | {phase2_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Accuracy Gain: {acc_improvement:+.2f}%")
    print(f"PPL Improvement: {ppl_improvement:+.2f}%")

    return {
        'model_name': model_name,
        'phase1_acc': phase1_acc,
        'phase1_ppl': phase1_ppl,
        'phase1_time': phase1_time,
        'phase2_acc': phase2_acc,
        'phase2_ppl': phase2_ppl,
        'phase2_time': phase2_time,
        'total_time': total_time,
        'acc_gain': acc_improvement,
        'ppl_gain': ppl_improvement,
        'phase1_history': result_phase1,
        'phase2_history': result_phase2,
    }


def main():
    print("="*60)
    print("Progressive Layer Training Experiment")
    print("="*60)
    print(f"Device: {get_device()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nExperiment Design:")
    print("  Phase 1: Train 3-layer model normally")
    print("  Phase 2: Add 1 layer → Freeze first 3 → Train only new layer")
    print()

    device = get_device()

    # Run experiment (Standard Transformer with final layer loss only)
    result = run_progressive_experiment(
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
