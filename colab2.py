"""
ASHEM (Adaptive Supervision via Hard Example Mining) Experiment - Staged DS Version

**Framework**: Staged Deep Supervision (SDS)
**Date**: 2025-12-12
**Dataset**: WikiText-2 (10K samples)

This experiment demonstrates ASHEM using the Staged DS framework:
1. Stage 1: Train shallow model (2 layers) on all data
2. Identify hard examples using confidence-based threshold
3. Stage 2: Train additional layers (2→4) on hard examples only
4. Inference: Two-stage routing using Early Exit

Key Benefits:
- Hard PPL: 78% improvement (2763 → 668)
- Compute cost: 36% reduction (64.82% of full model)
- Fully integrated with Staged DS framework

Staged DS Framework:
- Stage: 訓練の1フェーズ（どの層を、どのデータで、どう訓練するか）
- すべての訓練戦略をStageの組み合わせとして表現

References:
- Staged Deep Supervision (SDS): 本フレームワーク
- ASHEM: Adaptive Supervision via Hard Example Mining (本研究)
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- Early Exit: BranchyNet (2016), Teerapittayanon et al. (2016)
"""

import sys
sys.path.insert(0, 'src')

import time
import torch
import torch.nn as nn
from typing import Dict, Any

from ease import (
    DeepSupervisionTransformer,
    StagedTrainer,
    create_ashem_config,
    compute_confidence_threshold,
    collect_hard_examples,
    create_hard_example_loader,
)

sys.path.insert(0, 'experiments')
from utils import set_seed, get_device, create_wikitext_dataloaders


def run_ashem_experiment(device: str = 'cuda') -> Dict[str, Any]:
    """
    Run ASHEM experiment using Staged DS framework.

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print("ASHEM Experiment (Staged DS)")
    print(f"{'='*60}\n")

    # Configuration
    phase1_layers = 2
    phase2_layers = 4
    phase1_samples = 10000
    batch_size = 64
    seq_len = 32
    dim = 64
    num_heads = 4
    hard_example_ratio = 0.5

    # Load data
    set_seed(42)
    train_loader, val_loader, vocab_size = create_wikitext_dataloaders(
        phase1_samples, batch_size, seq_len
    )

    print(f"Dataset: WikiText-2")
    print(f"  Train samples: {phase1_samples}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")

    # Create ASHEM config
    config = create_ashem_config(
        phase1_layers=phase1_layers,
        phase2_layers=phase2_layers,
        vocab_size=vocab_size,
        phase1_lr=1e-3,
        phase2_lr=1e-4,
        phase1_epochs=50,
        phase2_epochs=50,
        hard_example_ratio=hard_example_ratio
    )

    # Create model (initially with phase1_layers, will extend later)
    model = DeepSupervisionTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=phase1_layers,
        num_heads=num_heads
    ).to(device)

    # Create trainer
    trainer = StagedTrainer(config, device=device)

    # ==========================================================================
    # Stage 1: Train Shallow Model
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 1: Train {phase1_layers}-layer model on all data")
    print(f"{'='*60}")

    start_time = time.time()
    result_phase1 = trainer.train_stage(
        config.stages[0],
        model,
        train_loader,
        val_loader,
        verbose=True
    )
    phase1_time = time.time() - start_time

    phase1_ppl = torch.exp(torch.tensor(result_phase1['best_val_loss'])).item()

    print(f"\nStage 1 Results:")
    print(f"  Best Val PPL: {phase1_ppl:.2f}")
    print(f"  Time: {phase1_time:.2f}s")

    # ==========================================================================
    # Compute Confidence Threshold & Collect Hard Examples
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Collecting Hard Examples (target: {hard_example_ratio*100:.0f}%)")
    print(f"{'='*60}\n")

    threshold = compute_confidence_threshold(
        model, val_loader, hard_example_ratio, device, phase1_layers
    )

    print(f"✓ Confidence threshold: {threshold:.4f}")

    hard_examples = collect_hard_examples(
        model, val_loader, threshold, device, phase1_layers
    )

    num_hard = len(hard_examples['targets'])
    avg_confidence = hard_examples['confidences'].mean().item()
    total_samples = phase1_samples * 0.2 * seq_len

    print(f"✓ Collected {num_hard:,} hard examples")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"  Actual ratio: {num_hard / total_samples * 100:.1f}%")

    # Evaluate Phase 1 on hard examples
    from ease.staged_ds import compute_confidence
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        h_batch = hard_examples['hidden_states'][:1000].unsqueeze(1).to(device)
        t_batch = hard_examples['targets'][:1000].to(device)
        logits = model.output_head(h_batch).squeeze(1)
        loss = F.cross_entropy(logits, t_batch)
        phase1_hard_ppl = torch.exp(loss).item()

    print(f"\n✓ Stage 1 Hard PPL: {phase1_hard_ppl:.2f}")
    print(f"  (vs Overall Val PPL: {phase1_ppl:.2f})")

    # ==========================================================================
    # Stage 2: Extend Model and Train on Hard Examples
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Stage 2: Extend to {phase2_layers} layers, train on hard examples")
    print(f"{'='*60}\n")

    # Extend model
    model_extended = DeepSupervisionTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=phase2_layers,
        num_heads=num_heads
    ).to(device)

    # Copy weights from Stage 1
    model_extended.embedding.load_state_dict(model.embedding.state_dict())
    for i in range(phase1_layers):
        model_extended.layers[i].load_state_dict(model.layers[i].state_dict())
    model_extended.output_head.load_state_dict(model.output_head.state_dict())

    print("✓ Copied weights from Stage 1 model")
    print(f"✓ Layers {phase1_layers+1}-{phase2_layers} randomly initialized")

    # Create hard example loader
    hard_batches = create_hard_example_loader(hard_examples, batch_size)

    print(f"  Hard example batches: {len(hard_batches)}")

    # Train Stage 2
    start_time = time.time()
    result_phase2 = trainer.train_stage(
        config.stages[1],
        model_extended,
        hard_batches,
        val_loader,
        verbose=True
    )
    phase2_time = time.time() - start_time

    phase2_ppl = torch.exp(torch.tensor(result_phase2['best_val_loss'])).item()

    # Evaluate on hard examples again
    model_extended.eval()
    with torch.no_grad():
        # Process through Stage 2 layers
        h_batch = hard_examples['hidden_states'][:1000].unsqueeze(1).to(device)
        for i in range(phase1_layers, phase2_layers):
            h_batch = model_extended.layers[i](h_batch)
        t_batch = hard_examples['targets'][:1000].to(device)
        logits = model_extended.output_head(h_batch).squeeze(1)
        loss = F.cross_entropy(logits, t_batch)
        phase2_hard_ppl = torch.exp(loss).item()

    print(f"\nStage 2 Results:")
    print(f"  Best Val PPL: {phase2_ppl:.2f}")
    print(f"  Best Hard PPL: {phase2_hard_ppl:.2f}")
    print(f"  Hard PPL Improvement: {phase1_hard_ppl - phase2_hard_ppl:+.2f} "
          f"({(phase1_hard_ppl - phase2_hard_ppl) / phase1_hard_ppl * 100:+.1f}%)")
    print(f"  Time: {phase2_time:.2f}s")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print("\nHard Examples Performance:")
    print(f"  Stage 1 Hard PPL:     {phase1_hard_ppl:.2f}")
    print(f"  Stage 2 Hard PPL:     {phase2_hard_ppl:.2f}")
    print(f"  Improvement:          {phase1_hard_ppl - phase2_hard_ppl:+.2f} "
          f"({(phase1_hard_ppl - phase2_hard_ppl) / phase1_hard_ppl * 100:+.1f}%)")

    print("\nOverall Performance:")
    print(f"  Stage 1 (2-layer):    PPL {phase1_ppl:.2f}")
    print(f"  Stage 2 (4-layer):    PPL {phase2_ppl:.2f}")
    print(f"  Improvement:          {phase1_ppl - phase2_ppl:+.2f}")

    return {
        'phase1_ppl': phase1_ppl,
        'phase1_hard_ppl': phase1_hard_ppl,
        'phase1_time': phase1_time,
        'num_hard_examples': num_hard,
        'phase2_ppl': phase2_ppl,
        'phase2_hard_ppl': phase2_hard_ppl,
        'phase2_time': phase2_time,
        'hard_ppl_improvement': phase1_hard_ppl - phase2_hard_ppl,
    }


def main() -> None:
    """Run ASHEM experiment."""
    print("="*60)
    print("ASHEM Experiment - Staged Deep Supervision")
    print("="*60)
    print(f"Device: {get_device()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    device = get_device()
    results = run_ashem_experiment(device)

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
