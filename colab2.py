"""
LEGO Experiment - Hard Example Mining with Two-Stage Inference

Workflow:
1. Phase 1: Train 2-layer model on all data
2. Collect hard examples (low confidence tokens)
3. Phase 2: Extend to 4 layers, train upper layers on hard examples only
4. Inference: Early Exit routing (Layer 2 for easy, Layer 4 for hard)
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
from typing import Dict, Any

from lego import (
    LEGOTransformer,
    Trainer,
    set_seed,
    get_device,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
    evaluate_on_hard_examples,
    create_wikitext_dataloaders,
    ExperimentConfig,
)


def run_experiment(config: ExperimentConfig, device: str) -> Dict[str, Any]:
    """Run complete hard example mining experiment."""
    print(f"\n{'='*60}")
    print("LEGOTransformer - Hard Example Mining")
    print(f"{'='*60}\n")

    # Phase 1: Train Shallow Model
    print(f"Phase 1: Train {config.phase1_layers}-layer model")
    print(f"{'='*60}")

    set_seed(42)
    train_loader, val_loader, vocab_size = create_wikitext_dataloaders(
        config.phase1_samples, config.phase1_batch, config.seq_len
    )

    model = LEGOTransformer(
        vocab_size=vocab_size,
        dim=config.dim,
        num_layers=config.phase1_layers,
        num_heads=config.num_heads
    ).to(device)

    trainer = Trainer(vocab_size=vocab_size, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.phase1_lr)

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
    phase1_ppl = result_phase1['val_ppls'][result_phase1['best_epoch']]

    print(f"\nPhase 1 Results: Acc {phase1_acc:.2f}% | PPL {phase1_ppl:.2f} | Time {phase1_time:.2f}s")

    # Compute Confidence Threshold
    print(f"\n{'='*60}")
    print(f"Computing Confidence Threshold (target: {config.hard_example_ratio*100:.0f}%)")
    print(f"{'='*60}\n")

    confidence_threshold = compute_confidence_threshold(
        model, val_loader, config.hard_example_ratio, device
    )
    print(f"Threshold: {confidence_threshold:.4f}")

    # Collect Hard Examples
    hard_examples = collect_hard_examples(model, val_loader, confidence_threshold, device)
    num_hard = len(hard_examples['targets'])
    total_samples = config.phase1_samples * 0.2 * config.seq_len
    print(f"Collected {num_hard:,} hard examples ({num_hard / total_samples * 100:.1f}%)")

    # Evaluate Phase 1 on Hard Examples
    phase1_hard_ppl = evaluate_on_hard_examples(
        model, hard_examples, device,
        batch_size=config.phase2_batch, num_lower_layers=config.phase1_layers
    )
    print(f"Phase 1 Hard PPL: {phase1_hard_ppl:.2f}")

    # Phase 2: Extend Model and Train on Hard Examples
    print(f"\n{'='*60}")
    print("Phase 2: Add 2 layers, train on hard examples")
    print(f"{'='*60}\n")

    model_extended = model.extend(
        num_layers=config.phase2_layers,
        routing_threshold=confidence_threshold,
        freeze_lower=True
    ).to(device)

    trainable = sum(p.numel() for p in model_extended.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_extended.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    hard_batches = create_hard_example_loader(hard_examples, config.phase2_batch)
    optimizer_upper = torch.optim.AdamW(model_extended.parameters(), lr=config.phase2_lr)

    start_time = time.time()
    result_phase2 = trainer.train_upper_layers_with_early_stopping(
        model=model_extended,
        hard_batches=hard_batches,
        val_batches=val_loader,
        hard_examples=hard_examples,
        optimizer=optimizer_upper,
        num_lower_layers=config.phase1_layers,
        routing_threshold=confidence_threshold,
        max_epochs=config.phase2_epochs,
        patience=config.phase2_patience,
        verbose=True
    )
    phase2_time = time.time() - start_time

    phase2_hard_ppl = result_phase2['hard_ppls'][result_phase2['best_epoch']]
    print(f"\nPhase 2 Results: Hard PPL {phase2_hard_ppl:.2f} | Time {phase2_time:.2f}s")
    print(f"Hard PPL Improvement: {phase1_hard_ppl - phase2_hard_ppl:+.2f} "
          f"({(phase1_hard_ppl - phase2_hard_ppl) / phase1_hard_ppl * 100:+.1f}%)")

    # Final Evaluation: Two-Stage Inference
    print(f"\n{'='*60}")
    print("Final Evaluation (Two-Stage Inference)")
    print(f"{'='*60}\n")

    stats = trainer.evaluate(model_extended, val_loader, routing_threshold=confidence_threshold)

    print(f"Accuracy: {stats['acc']*100:.2f}%")
    print(f"PPL: {stats['ppl']:.2f}")
    print(f"Shallow ratio: {stats['shallow_ratio']*100:.1f}%")
    print(f"Compute cost: {stats['compute_cost']:.2%}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Phase 1: Acc {phase1_acc:.2f}% | PPL {phase1_ppl:.2f}")
    print(f"Phase 2: Acc {stats['acc']*100:.2f}% | PPL {stats['ppl']:.2f}")
    print(f"Hard PPL: {phase1_hard_ppl:.2f} -> {phase2_hard_ppl:.2f}")

    return {
        'phase1_acc': phase1_acc,
        'phase1_ppl': phase1_ppl,
        'phase1_hard_ppl': phase1_hard_ppl,
        'phase2_hard_ppl': phase2_hard_ppl,
        'two_stage_acc': stats['acc'] * 100,
        'two_stage_ppl': stats['ppl'],
        'shallow_ratio': stats['shallow_ratio'],
        'compute_cost': stats['compute_cost']
    }


def main() -> None:
    """Run Hard Example Mining experiment."""
    config = ExperimentConfig()
    device = get_device()

    print("="*60)
    print("LEGO: Hard Example Mining + Two-Stage Inference")
    print("="*60)
    print(f"Device: {device}")

    run_experiment(config, device)

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
