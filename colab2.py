"""
Hard Example Mining + Two-Stage Inference Experiment

Experiment Design:
- Phase 1: Train 2-layer model + collect hard examples (low confidence)
- Phase 2: Train additional 2 layers on hard examples only
- Validation: Two-stage inference using EASE framework's Early Exit

Key Improvement: Uses EASE's built-in Early Exit functionality instead of
manual implementation for cleaner, more maintainable code.
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict

from ease import (
    StandardTransformer,
    DeepSupervisionTransformer,
    Trainer,
    TrainingConfig,
    create_standard_config,
)

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
    vocab_size: int = 69830
    seq_len: int = 32
    dim: int = 64
    num_heads: int = 4
    base_lr: float = 1e-3

    # Phase 1: Train 2-layer model
    phase1_layers: int = 2
    phase1_samples: int = 10000
    phase1_batch: int = 64
    phase1_epochs: int = 50
    phase1_patience: int = 1

    # Hard example threshold (auto-adjusted to target this percentage)
    hard_example_ratio: float = 0.5  # Target 50% of examples as hard examples

    # Phase 2: Train on hard examples
    phase2_layers: int = 4  # Total: 2 + 2
    phase2_batch: int = 64
    phase2_epochs: int = 50
    phase2_patience: int = 3  # Higher patience for randomly initialized layers
    phase2_lr: float = 1e-4  # Lower LR for fine-tuning


CONFIG = Config()


def compute_confidence(model: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Compute confidence (max probability) from hidden state.

    Centralized confidence computation to avoid duplication.

    Args:
        model: Model with output_head
        hidden_state: Hidden state tensor (batch_size, seq_len, dim)

    Returns:
        Confidence values (batch_size, seq_len)
    """
    logits = model.output_head(hidden_state)
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def compute_confidence_threshold(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    target_ratio: float,
    device: str
) -> float:
    """
    Compute confidence threshold to achieve target hard example ratio.

    Args:
        model: Trained model
        val_batches: Validation batches
        target_ratio: Target ratio of hard examples (e.g., 0.5 for 50%)
        device: Device to use

    Returns:
        Confidence threshold value
    """
    model.eval()
    all_confidences = []

    with torch.no_grad():
        for x, y in val_batches:
            x = x.to(device)

            # Get output from all layers
            h = model.embedding(x)
            for layer in model.layers:
                h = layer(h)

            # Compute confidence using centralized function
            confidence = compute_confidence(model, h)
            all_confidences.append(confidence.view(-1))

    # Concatenate all confidences
    all_confidences = torch.cat(all_confidences)

    # Compute threshold as the target_ratio percentile
    # For 50%, we want the median confidence value
    threshold = torch.quantile(all_confidences, target_ratio).item()

    return threshold


def collect_hard_examples(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Collect hard examples (low confidence samples) from validation set.

    Returns:
        Dictionary with:
        - 'inputs': Input sequences
        - 'hidden_states': Layer 2 outputs (hidden states)
        - 'targets': Target labels
        - 'confidences': Confidence scores
    """
    model.eval()

    hard_inputs = []
    hard_hidden_states = []
    hard_targets = []
    hard_confidences = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)

            # Get output from all layers
            h = model.embedding(x)
            for layer in model.layers:
                h = layer(h)

            # Compute confidence using centralized function
            confidence = compute_confidence(model, h)

            # Collect low-confidence samples
            mask = confidence < threshold

            if mask.any():
                # Flatten batch and sequence dimensions
                x_flat = x.view(-1)
                h_flat = h.view(-1, h.shape[-1])
                y_flat = y.view(-1)
                confidence_flat = confidence.view(-1)
                mask_flat = mask.view(-1)

                # Collect hard examples
                hard_inputs.append(x_flat[mask_flat])
                hard_hidden_states.append(h_flat[mask_flat])
                hard_targets.append(y_flat[mask_flat])
                hard_confidences.append(confidence_flat[mask_flat])

    if not hard_inputs:
        raise ValueError("No hard examples found. Try increasing threshold.")

    return {
        'inputs': torch.cat(hard_inputs),
        'hidden_states': torch.cat(hard_hidden_states),
        'targets': torch.cat(hard_targets),
        'confidences': torch.cat(hard_confidences)
    }


def create_hard_example_loader(
    hard_examples: Dict[str, torch.Tensor],
    batch_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create dataloader from hard examples."""
    hidden_states = hard_examples['hidden_states']  # Shape: (num_samples, dim)
    targets = hard_examples['targets']

    num_samples = len(targets)
    indices = torch.randperm(num_samples)

    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        # Add seq_len dimension: (batch_size, dim) → (batch_size, 1, dim)
        h_batch = hidden_states[batch_indices].unsqueeze(1)
        t_batch = targets[batch_indices]
        batches.append((h_batch, t_batch))

    return batches


def evaluate_on_hard_examples(
    model: nn.Module,
    hard_examples: Dict[str, torch.Tensor],
    vocab_size: int,
    device: str,
    batch_size: int = 64,
    num_lower_layers: int = 2
) -> float:
    """
    Evaluate model on hard examples only (measure PPL on hard examples).

    Args:
        model: Model to evaluate (can be 2-layer or 4-layer)
        hard_examples: Dictionary with 'hidden_states' and 'targets'
        vocab_size: Vocabulary size for loss computation
        device: Device to use
        batch_size: Batch size for evaluation
        num_lower_layers: Number of lower layers (if 4-layer model, process through layers 3-4)

    Returns:
        Perplexity on hard examples
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    hidden_states = hard_examples['hidden_states']
    targets = hard_examples['targets']
    num_samples = len(targets)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Get batch
            h_batch = hidden_states[i:i + batch_size].unsqueeze(1).to(device)  # (batch_size, 1, dim)
            y_batch = targets[i:i + batch_size].to(device)

            # If 4-layer model, process through upper layers (3-4)
            if hasattr(model, 'num_layers') and model.num_layers > num_lower_layers:
                for layer_idx in range(num_lower_layers, model.num_layers):
                    h_batch = model.layers[layer_idx](h_batch)

            # Compute loss
            logits = model.output_head(h_batch).squeeze(1)  # (batch_size, vocab_size)
            loss = F.cross_entropy(logits, y_batch, reduction='sum')

            total_loss += loss.item()
            total_samples += len(y_batch)

    avg_loss = total_loss / total_samples
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def train_upper_layers(
    model: nn.Module,
    hard_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str,
    num_lower_layers: int = 2
) -> float:
    """
    Train upper layers (3-4) on hard examples.

    Args:
        model: Extended model with upper layers
        hard_batches: Batches of hard examples (hidden states, targets)
        optimizer: Optimizer for upper layers
        vocab_size: Vocabulary size for loss computation
        device: Device to use
        num_lower_layers: Number of lower layers (already trained)

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for h, y in hard_batches:
        h, y = h.to(device), y.to(device)
        optimizer.zero_grad()

        # Process through upper layers only
        for i in range(num_lower_layers, model.num_layers):
            h = model.layers[i](h)

        # Compute loss
        # h shape: (batch_size, 1, dim)
        # Remove seq_len dimension for classification
        logits = model.output_head(h).squeeze(1)  # (batch_size, vocab_size)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(hard_batches)


def run_experiment(model_name: str, ModelClass, config_fn, device: str):
    """Run hard example mining experiment."""
    print(f"\n{'='*60}")
    print(f"{model_name} - Hard Example Mining")
    print(f"{'='*60}\n")

    # ========================================
    # Phase 1: Train 2-layer model
    # ========================================
    print(f"Phase 1: Train {CONFIG.phase1_layers}-layer model")
    print(f"{'='*60}")

    set_seed(42)
    train_loader, val_loader = create_dataloaders_from_colab(
        CONFIG.phase1_samples, CONFIG.phase1_batch, CONFIG.seq_len
    )

    # Create 2-layer model
    model = ModelClass(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.phase1_layers,
        num_heads=CONFIG.num_heads
    ).to(device)

    config = config_fn(num_layers=CONFIG.phase1_layers)
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
    # Compute Confidence Threshold
    # ========================================
    print(f"\n{'='*60}")
    print(f"Computing Confidence Threshold (target ratio: {CONFIG.hard_example_ratio*100:.0f}%)")
    print(f"{'='*60}\n")

    confidence_threshold = compute_confidence_threshold(
        model, val_loader, CONFIG.hard_example_ratio, device
    )

    print(f"✓ Computed confidence threshold: {confidence_threshold:.4f}")
    print(f"  Examples with confidence < {confidence_threshold:.4f} will be treated as hard examples")

    # ========================================
    # Collect Hard Examples
    # ========================================
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
    print(f"  Actual ratio: {num_hard / total_samples * 100:.1f}% (target: {CONFIG.hard_example_ratio*100:.0f}%)")

    # ========================================
    # Evaluate Phase 1 on Hard Examples
    # ========================================
    print(f"\n{'='*60}")
    print("Evaluating Phase 1 on Hard Examples")
    print(f"{'='*60}\n")

    phase1_hard_ppl = evaluate_on_hard_examples(
        model, hard_examples, CONFIG.vocab_size, device,
        batch_size=CONFIG.phase2_batch, num_lower_layers=CONFIG.phase1_layers
    )

    print(f"✓ Phase 1 Hard PPL: {phase1_hard_ppl:.2f}")
    print(f"  (vs Overall Val PPL: {phase1_ppl:.2f})")

    # ========================================
    # Phase 2: Extend to 4 layers + Train on hard examples
    # ========================================
    print(f"\n{'='*60}")
    print("Phase 2: Add 2 layers → Train on hard examples")
    print(f"{'='*60}\n")

    # Create 4-layer model with Early Exit support
    # Use DeepSupervisionTransformer for Phase 2 (supports Early Exit)
    model_extended = DeepSupervisionTransformer(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.phase2_layers,
        num_heads=CONFIG.num_heads,
        exit_layer=CONFIG.phase1_layers,  # Exit at Layer 2
        routing_threshold=confidence_threshold  # Use auto-computed threshold
    ).to(device)

    # Copy lower layers (1-2) weights from Phase 1 model
    model_extended.embedding.load_state_dict(model.embedding.state_dict())
    for i in range(CONFIG.phase1_layers):
        model_extended.layers[i].load_state_dict(model.layers[i].state_dict())
    model_extended.output_head.load_state_dict(model.output_head.state_dict())

    print("✓ Copied weights from 2-layer model")
    print("✓ Layers 3-4 randomly initialized")

    # Freeze lower layers (1-2)
    for param in model_extended.embedding.parameters():
        param.requires_grad = False
    for i in range(CONFIG.phase1_layers):
        for param in model_extended.layers[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model_extended.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_extended.parameters())
    print("✓ Frozen lower 2 layers")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Create hard example loader
    hard_batches = create_hard_example_loader(hard_examples, CONFIG.phase2_batch)
    print(f"  Hard example batches: {len(hard_batches)}")

    # Train upper layers
    optimizer_upper = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_extended.parameters()),
        lr=CONFIG.phase2_lr
    )

    print(f"  Using learning rate: {CONFIG.phase2_lr:.1e}")

    best_val_ppl = float('inf')
    best_model_state = None
    patience_counter = 0

    start_time = time.time()
    for epoch in range(CONFIG.phase2_epochs):
        # Train on hard examples
        train_loss = train_upper_layers(
            model_extended, hard_batches, optimizer_upper,
            CONFIG.vocab_size, device, CONFIG.phase1_layers
        )

        # Evaluate with EASE's built-in Early Exit evaluation
        # Create temporary trainer with Early Exit config
        eval_config = TrainingConfig(
            layer_weights={i: 0 for i in range(1, CONFIG.phase2_layers + 1)},
            routing_threshold=confidence_threshold,
            exit_layer=CONFIG.phase1_layers
        )
        eval_config.layer_weights[CONFIG.phase2_layers] = 1.0  # Set final layer weight

        eval_trainer = Trainer(eval_config, vocab_size=CONFIG.vocab_size, device=device)
        val_stats = eval_trainer.evaluate(model_extended, val_loader)

        val_acc = val_stats['acc']
        val_ppl = val_stats['ppl']
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Evaluate on hard examples
        hard_ppl = evaluate_on_hard_examples(
            model_extended, hard_examples, CONFIG.vocab_size, device,
            batch_size=CONFIG.phase2_batch, num_lower_layers=CONFIG.phase1_layers
        )

        print(f"Epoch {epoch+1}/{CONFIG.phase2_epochs} - "
              f"Train PPL: {train_ppl:.4f} | "
              f"Val PPL: {val_ppl:.2f} | "
              f"Val Acc: {val_acc*100:.2f}% | "
              f"Hard PPL: {hard_ppl:.2f}")

        # Early stopping based on validation PPL
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_model_state = {k: v.cpu().clone() for k, v in model_extended.state_dict().items()}
            patience_counter = 0
            print(f"  → New best (val_ppl: {val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{CONFIG.phase2_patience})")

        if patience_counter >= CONFIG.phase2_patience:
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
        batch_size=CONFIG.phase2_batch, num_lower_layers=CONFIG.phase1_layers
    )

    print("\nPhase 2 Results:")
    print(f"  Best Val PPL: {best_val_ppl:.2f}")
    print(f"  Best Hard PPL: {phase2_hard_ppl:.2f}")
    print(f"  Hard PPL Improvement: {phase1_hard_ppl - phase2_hard_ppl:+.2f} ({(phase1_hard_ppl - phase2_hard_ppl) / phase1_hard_ppl * 100:+.1f}%)")
    print(f"  Time: {phase2_time:.2f}s")

    # ========================================
    # Final Evaluation (Two-Stage Inference with EASE)
    # ========================================
    print(f"\n{'='*60}")
    print("Final Evaluation (Two-Stage Inference)")
    print(f"{'='*60}\n")

    # Use EASE framework's built-in Early Exit evaluation
    final_config = TrainingConfig(
        layer_weights={i: 0 for i in range(1, CONFIG.phase2_layers + 1)},
        routing_threshold=confidence_threshold,
        exit_layer=CONFIG.phase1_layers
    )
    final_config.layer_weights[CONFIG.phase2_layers] = 1.0  # Set final layer weight

    final_trainer = Trainer(final_config, vocab_size=CONFIG.vocab_size, device=device)
    stats = final_trainer.evaluate(model_extended, val_loader)

    print("Results:")
    print(f"  Accuracy: {stats['acc']*100:.2f}%")
    print(f"  Shallow ratio (Layer 2): {stats['shallow_ratio']*100:.1f}%")
    print(f"  Deep ratio (Layer 4): {(1-stats['shallow_ratio'])*100:.1f}%")
    print(f"  Compute cost: {stats['compute_cost']:.2%} of full model")

    # Compare with baseline
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
    print(f"  Hard PPL Improvement:    {phase1_hard_ppl - phase2_hard_ppl:+.2f} ({(phase1_hard_ppl - phase2_hard_ppl) / phase1_hard_ppl * 100:+.1f}%)")

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


def main():
    print("="*60)
    print("Hard Example Mining + Two-Stage Inference")
    print("="*60)
    print(f"Device: {get_device()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nExperiment Design:")
    print(f"  Phase 1: Train {CONFIG.phase1_layers}-layer model")
    print(f"  Compute: Auto-adjust threshold to collect {CONFIG.hard_example_ratio*100:.0f}% hard examples")
    print(f"  Phase 2: Add {CONFIG.phase2_layers - CONFIG.phase1_layers} layers → Train on hard examples")
    print("  Eval: Two-stage inference (Layer 2 or Layer 4) using EASE's Early Exit")
    print()

    device = get_device()

    # Run experiment (Standard Transformer with final layer loss only)
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
