"""
LEGO Experiment - TRUE Early Exit Generation

Workflow:
1. Phase 1: Train Block 1 (2-layer model) on all data
2. Collect hard examples (low confidence tokens)
3. Phase 2: Extend with Block 2 (2 layers), train on hard examples only
4. TRUE Early Exit generation - actually skip Block 2 for easy tokens
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
    split_hard_examples,
    create_wikitext_dataloaders,
    ExperimentConfig,
)


def run_experiment(config: ExperimentConfig, device: str) -> Dict[str, Any]:
    """Run experiment with TRUE early exit generation."""
    print(f"\n{'='*60}")
    print("LEGOTransformer - TRUE Early Exit")
    print(f"{'='*60}\n")

    # Phase 1: Train Block 1
    print(f"Phase 1: Train Block 1 ({config.phase1_layers}-layer model)")
    print(f"{'='*60}")

    set_seed(42)
    train_loader, val_loader, vocab_size = create_wikitext_dataloaders(
        config.phase1_samples, config.phase1_batch, config.seq_len
    )

    model = LEGOTransformer.create(
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
        model, val_loader, config.hard_example_ratio, device, block_idx=0
    )
    print(f"Threshold: {confidence_threshold:.4f}")

    # Collect Hard Examples and split into train/val
    hard_examples = collect_hard_examples(model, val_loader, confidence_threshold, device, block_idx=0)
    num_hard = len(hard_examples['targets'])
    total_samples = config.phase1_samples * 0.2 * config.seq_len
    print(f"Collected {num_hard:,} hard examples ({num_hard / total_samples * 100:.1f}%)")

    # Split hard examples for independent train/val
    hard_train, hard_val = split_hard_examples(hard_examples, train_ratio=0.8)
    print(f"Hard train: {len(hard_train['targets']):,} | Hard val: {len(hard_val['targets']):,}")

    # Phase 2: Extend Model with Block 2
    num_new_layers = config.phase2_layers - config.phase1_layers
    print(f"\n{'='*60}")
    print(f"Phase 2: Add Block 2 ({num_new_layers} layers), train on hard examples")
    print(f"{'='*60}\n")

    model_extended = model.extend(
        num_new_layers=num_new_layers,
        threshold=confidence_threshold,
        freeze_existing=True
    ).to(device)

    trainable = sum(p.numel() for p in model_extended.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_extended.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"Blocks: {len(model_extended.blocks)} (layers per block: {[b.num_layers for b in model_extended.blocks]})")

    optimizer_block2 = torch.optim.AdamW(model_extended.parameters(), lr=config.phase2_lr)

    start_time = time.time()
    result_phase2 = trainer.train_block(
        model=model_extended,
        hard_train=hard_train,
        hard_val=hard_val,
        optimizer=optimizer_block2,
        start_block_idx=1,  # Train Block 2 (index 1)
        batch_size=config.phase2_batch,
        max_epochs=config.phase2_epochs,
        patience=config.phase2_patience,
        verbose=True
    )
    phase2_time = time.time() - start_time

    phase2_val_ppl = result_phase2['val_ppls'][result_phase2['best_epoch']]
    print(f"\nPhase 2 Results: Val PPL (hard) {phase2_val_ppl:.2f} | Time {phase2_time:.2f}s")

    # Final Evaluation with routing (uses block thresholds)
    print(f"\n{'='*60}")
    print("Evaluation: Routing (both paths computed for comparison)")
    print(f"{'='*60}\n")

    stats_routing = trainer.evaluate(model_extended, val_loader, use_routing=True)

    print(f"Accuracy: {stats_routing['acc']*100:.2f}%")
    print(f"PPL: {stats_routing['ppl']:.2f}")
    print(f"Shallow ratio: {stats_routing['shallow_ratio']*100:.1f}%")
    print(f"Compute cost (theoretical): {stats_routing['compute_cost']:.2%}")

    # TRUE Early Exit Generation Demo
    print(f"\n{'='*60}")
    print("TRUE Early Exit Generation Demo")
    print(f"{'='*60}\n")

    # Use first batch as prompt
    prompt_batch = val_loader[0][0][:1, :8].to(device)  # batch=1, seq=8
    max_new_tokens = 32

    print(f"Prompt length: {prompt_batch.shape[1]} tokens")
    print(f"Generating: {max_new_tokens} new tokens")
    print(f"Blocks: {len(model_extended.blocks)} (layers: {[b.num_layers for b in model_extended.blocks]})")
    print(f"Block 1 threshold: {model_extended.blocks[0].threshold:.4f}")

    # Generate with TRUE early exit
    set_seed(123)
    start_time = time.time()
    generated, early_exit_stats = model_extended.generate(
        prompt_batch,
        max_new_tokens=max_new_tokens,
        temperature=1.0
    )
    gen_time = time.time() - start_time

    print(f"\nGeneration Results:")
    print(f"  Time: {gen_time:.4f}s")
    print(f"  Exit counts: {early_exit_stats['exit_counts']}")
    print(f"  Shallow ratio: {early_exit_stats['shallow_ratio']:.1%}")
    print(f"  Compute cost: {early_exit_stats['compute_cost']:.1%}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Phase 1: Acc {phase1_acc:.2f}% | PPL {phase1_ppl:.2f}")
    print(f"Phase 2 (hard val): PPL {phase2_val_ppl:.2f}")
    print(f"Final (full val): Acc {stats_routing['acc']*100:.2f}% | PPL {stats_routing['ppl']:.2f}")
    print(f"\nTRUE Early Exit Stats:")
    print(f"  Shallow ratio: {early_exit_stats['shallow_ratio']:.1%}")
    print(f"  Compute cost: {early_exit_stats['compute_cost']:.1%}")
    print(f"  Compute savings: {(1 - early_exit_stats['compute_cost'])*100:.1f}%")

    return {
        'phase1_acc': phase1_acc,
        'phase1_ppl': phase1_ppl,
        'phase2_val_ppl': phase2_val_ppl,
        'two_stage_acc': stats_routing['acc'] * 100,
        'two_stage_ppl': stats_routing['ppl'],
        'routing_shallow_ratio': stats_routing['shallow_ratio'],
        'routing_compute_cost': stats_routing['compute_cost'],
        'true_shallow_ratio': early_exit_stats['shallow_ratio'],
        'true_compute_cost': early_exit_stats['compute_cost'],
    }


def main() -> None:
    """Run TRUE Early Exit experiment."""
    config = ExperimentConfig()
    device = get_device()

    print("="*60)
    print("LEGO: TRUE Early Exit Generation")
    print("="*60)
    print(f"Device: {device}")

    run_experiment(config, device)

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
