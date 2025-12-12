"""
Hard Example Mining + Two-Stage Inference Experiment

Experiment Design:
- Phase 1: Train 2-layer model + collect hard examples (low confidence)
- Phase 2: Train additional 2 layers on hard examples only
- Validation: Two-stage inference
  - High confidence → use Layer 2 output
  - Low confidence → process through Layer 3-4
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
    create_standard_config,
    create_deep_supervision_config,
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

    # Hard example threshold
    confidence_threshold: float = 0.8  # Collect examples with confidence < 0.8

    # Phase 2: Train on hard examples
    phase2_layers: int = 4  # Total: 2 + 2
    phase2_batch: int = 64
    phase2_epochs: int = 50
    phase2_patience: int = 1
    phase2_lr: float = 1e-4  # Lower LR for fine-tuning

CONFIG = Config()


def create_dataloaders(num_samples: int, batch_size: int, seq_len: int):
    """Create train and validation dataloaders."""
    train_loader, val_loader = create_dataloaders_from_colab(num_samples, batch_size, seq_len)
    return train_loader, val_loader


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

            # Get Layer 2 output
            h = model.embedding(x)
            for i in range(model.num_layers):
                h = model.layers[i](h)

            # Compute output and confidence
            logits = model.output_head(h)
            probs = F.softmax(logits, dim=-1)
            confidence, preds = probs.max(dim=-1)

            # Collect low-confidence samples
            mask = confidence < threshold

            if mask.any():
                # Flatten batch and sequence dimensions
                batch_size, seq_len = x.shape
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
    hidden_states = hard_examples['hidden_states']
    targets = hard_examples['targets']

    num_samples = len(targets)
    indices = torch.randperm(num_samples)

    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batches.append((
            hidden_states[batch_indices],
            targets[batch_indices]
        ))

    return batches


def train_upper_layers(
    model: nn.Module,
    hard_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str,
    num_lower_layers: int = 2
) -> float:
    """Train upper layers (3-4) on hard examples."""
    model.train()
    total_loss = 0.0

    for h, y in hard_batches:
        h, y = h.to(device), y.to(device)
        optimizer.zero_grad()

        # Process through upper layers only
        for i in range(num_lower_layers, model.num_layers):
            h = model.layers[i](h)

        # Compute loss
        logits = model.output_head(h)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(hard_batches)


def evaluate_two_stage(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    vocab_size: int,
    device: str,
    num_lower_layers: int = 2
) -> Dict[str, float]:
    """
    Two-stage inference evaluation.

    Stage 1: Use Layer 2 output if confidence >= threshold
    Stage 2: Use Layer 4 output if confidence < threshold
    """
    model.eval()

    total_correct = 0
    total_tokens = 0
    total_shallow = 0
    total_deep = 0

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            batch_size, seq_len = x.shape

            # Stage 1: Process through lower layers (1-2)
            h = model.embedding(x)
            for i in range(num_lower_layers):
                h = model.layers[i](h)

            # Compute confidence
            logits_shallow = model.output_head(h)
            probs = F.softmax(logits_shallow, dim=-1)
            confidence, preds_shallow = probs.max(dim=-1)

            # Stage 2: Process low-confidence samples through upper layers (3-4)
            mask_deep = confidence < threshold

            if mask_deep.any():
                h_deep = h.clone()
                for i in range(num_lower_layers, model.num_layers):
                    h_deep = model.layers[i](h_deep)
                logits_deep = model.output_head(h_deep)
                preds_deep = logits_deep.argmax(dim=-1)

                # Combine predictions
                preds = torch.where(mask_deep, preds_deep, preds_shallow)
            else:
                preds = preds_shallow

            # Compute accuracy
            correct = int((preds == y).sum().item())
            total_correct += correct
            total_tokens += y.numel()

            # Count shallow/deep routing
            total_shallow += int((~mask_deep).sum().item())
            total_deep += int(mask_deep.sum().item())

    accuracy = total_correct / total_tokens
    shallow_ratio = total_shallow / total_tokens
    compute_cost = (total_shallow * num_lower_layers + total_deep * model.num_layers) / (total_tokens * model.num_layers)

    return {
        'accuracy': accuracy,
        'shallow_ratio': shallow_ratio,
        'deep_ratio': 1 - shallow_ratio,
        'compute_cost': compute_cost
    }


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
    train_loader, val_loader = create_dataloaders(
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
    # Collect Hard Examples
    # ========================================
    print(f"\n{'='*60}")
    print(f"Collecting Hard Examples (confidence < {CONFIG.confidence_threshold})")
    print(f"{'='*60}\n")

    hard_examples = collect_hard_examples(
        model, val_loader, CONFIG.confidence_threshold, device
    )

    num_hard = len(hard_examples['targets'])
    avg_confidence = hard_examples['confidences'].mean().item()

    print(f"✓ Collected {num_hard:,} hard examples")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"  Percentage of total: {num_hard / (CONFIG.phase1_samples * 0.2 * CONFIG.seq_len) * 100:.1f}%")

    # ========================================
    # Phase 2: Extend to 4 layers + Train on hard examples
    # ========================================
    print(f"\n{'='*60}")
    print("Phase 2: Add 2 layers → Train on hard examples")
    print(f"{'='*60}\n")

    # Create 4-layer model and copy weights
    model_extended = ModelClass(
        vocab_size=CONFIG.vocab_size,
        dim=CONFIG.dim,
        num_layers=CONFIG.phase2_layers,
        num_heads=CONFIG.num_heads
    ).to(device)

    # Copy lower layers (1-2)
    model_extended.embedding.load_state_dict(model.embedding.state_dict())
    for i in range(CONFIG.phase1_layers):
        model_extended.layers[i].load_state_dict(model.layers[i].state_dict())
    model_extended.output_head.load_state_dict(model.output_head.state_dict())

    print("✓ Copied weights from 2-layer model")
    print("✓ Layers 3-4 randomly initialized")

    # Freeze lower layers
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

    best_loss = float('inf')
    patience_counter = 0

    start_time = time.time()
    for epoch in range(CONFIG.phase2_epochs):
        train_loss = train_upper_layers(
            model_extended, hard_batches, optimizer_upper,
            CONFIG.vocab_size, device, CONFIG.phase1_layers
        )

        print(f"Epoch {epoch+1}/{CONFIG.phase2_epochs} - Train Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            print(f"  → New best (loss: {train_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{CONFIG.phase2_patience})")

        if patience_counter >= CONFIG.phase2_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    phase2_time = time.time() - start_time

    print("\nPhase 2 Results:")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Time: {phase2_time:.2f}s")

    # ========================================
    # Evaluate Two-Stage Inference
    # ========================================
    print(f"\n{'='*60}")
    print("Two-Stage Inference Evaluation")
    print(f"{'='*60}\n")

    stats = evaluate_two_stage(
        model_extended, val_loader, CONFIG.confidence_threshold,
        CONFIG.vocab_size, device, CONFIG.phase1_layers
    )

    print("Results:")
    print(f"  Accuracy: {stats['accuracy']*100:.2f}%")
    print(f"  Shallow ratio (Layer 2): {stats['shallow_ratio']*100:.1f}%")
    print(f"  Deep ratio (Layer 4): {stats['deep_ratio']*100:.1f}%")
    print(f"  Compute cost: {stats['compute_cost']:.2%} of full model")

    # Compare with baseline
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")
    print(f"Phase 1 (2-layer only):  {phase1_acc:.2f}%")
    print(f"Two-stage inference:     {stats['accuracy']*100:.2f}%")
    print(f"Improvement:             {stats['accuracy']*100 - phase1_acc:+.2f}%")
    print(f"Compute cost:            {stats['compute_cost']:.2%}")

    return {
        'model_name': model_name,
        'phase1_acc': phase1_acc,
        'phase1_time': phase1_time,
        'num_hard_examples': num_hard,
        'phase2_time': phase2_time,
        'two_stage_acc': stats['accuracy'] * 100,
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
    print(f"  Collect: Hard examples (confidence < {CONFIG.confidence_threshold})")
    print(f"  Phase 2: Add {CONFIG.phase2_layers - CONFIG.phase1_layers} layers → Train on hard examples")
    print("  Eval: Two-stage inference (Layer 2 or Layer 4)")
    print()

    device = get_device()
    results = []

    # Standard Transformer
    result = run_experiment(
        "Standard",
        StandardTransformer,
        create_standard_config,
        device
    )
    results.append(result)

    # Deep Supervision Transformer
    result = run_experiment(
        "Deep Supervision",
        DeepSupervisionTransformer,
        create_deep_supervision_config,
        device
    )
    results.append(result)

    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'2-layer':<12} {'Two-stage':<12} {'Improvement':<12} {'Compute':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<20} {r['phase1_acc']:>10.2f}% "
              f"{r['two_stage_acc']:>10.2f}% {r['two_stage_acc'] - r['phase1_acc']:>+10.2f}% "
              f"{r['compute_cost']:>10.1%}")
    print("="*80)

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
