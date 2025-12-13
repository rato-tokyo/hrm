"""
LEGO Framework Example - Block-wise Training with TRUE Early Exit

Workflow:
1. Create LEGOTransformer with multiple blocks
2. Train Block 0 on all data, collect hard examples
3. Train Block 1 on hard examples only
4. Evaluate with TRUE Early Exit
"""

import sys
sys.path.insert(0, 'src')

import torch
from lego import (
    LEGOTransformer,
    TrainingData,
    Trainer,
    ExperimentConfig,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
)


def main() -> None:
    """Run LEGO block-wise training example."""
    config = ExperimentConfig()
    device = get_device()

    print("=" * 60)
    print("LEGO Framework - Block-wise Training Example")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: dim={config.dim}, heads={config.num_heads}")

    # Setup
    set_seed(42)
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        config.phase1_samples, config.phase1_batch, config.seq_len
    )

    # Create model with 2 blocks
    # Block 0: threshold=0.8 (easy tokens exit here)
    # Block 1: threshold=1.0 (all remaining tokens processed)
    model = LEGOTransformer.create(
        vocab_size=vocab_size,
        dim=config.dim,
        num_heads=config.num_heads,
        layers_per_block=[config.phase1_layers, config.phase2_layers - config.phase1_layers],
        thresholds=[0.8, 1.0]
    ).to(device)

    print(f"Blocks: {len(model.blocks)}")
    print(f"Layers per block: {[b.num_layers for b in model.blocks]}")
    print(f"Thresholds: {[b.threshold for b in model.blocks]}")

    # Phase 1: Train Block 0 on all data
    print(f"\n{'=' * 60}")
    print("Phase 1: Train Block 0 on all data")
    print("=" * 60)

    # Create initial TrainingData from train batches
    # Embed tokens and flatten to (num_tokens, dim)
    all_hidden = []
    all_targets = []
    with torch.no_grad():
        for x, y in train_batches:
            x, y = x.to(device), y.to(device)
            h = model.embedding(x)  # (batch, seq, dim)
            all_hidden.append(h.view(-1, config.dim))
            all_targets.append(y.view(-1))

    initial_data = TrainingData(
        torch.cat(all_hidden),
        torch.cat(all_targets)
    )
    print(f"Initial data: {len(initial_data)} tokens")

    # Train Block 0
    optimizer0 = torch.optim.AdamW(model.blocks[0].parameters(), lr=config.phase1_lr)
    hard_data, stats0 = model.blocks[0].train_block(
        data=initial_data,
        optimizer=optimizer0,
        batch_size=config.phase1_batch,
        max_epochs=config.phase1_epochs,
        patience=config.phase1_patience,
        verbose=True
    )

    print(f"\nBlock 0 Results:")
    print(f"  Best PPL: {stats0['best_val_ppl']:.2f}")
    print(f"  Hard examples: {len(hard_data)} ({stats0['hard_ratio']*100:.1f}%)")

    # Phase 2: Train Block 1 on hard examples only
    print(f"\n{'=' * 60}")
    print("Phase 2: Train Block 1 on hard examples")
    print("=" * 60)

    if len(hard_data) > 0:
        optimizer1 = torch.optim.AdamW(model.blocks[1].parameters(), lr=config.phase2_lr)
        _, stats1 = model.blocks[1].train_block(
            data=hard_data,
            optimizer=optimizer1,
            batch_size=config.phase2_batch,
            max_epochs=config.phase2_epochs,
            patience=config.phase2_patience,
            verbose=True
        )
        print(f"\nBlock 1 Results:")
        print(f"  Best PPL: {stats1['best_val_ppl']:.2f}")
    else:
        print("No hard examples - Block 1 training skipped")

    # Evaluation with TRUE Early Exit
    print(f"\n{'=' * 60}")
    print("Evaluation: TRUE Early Exit")
    print("=" * 60)

    trainer = Trainer(vocab_size=vocab_size, device=device)
    stats = trainer.evaluate(model, val_batches, use_routing=True)

    print(f"\nFinal Results:")
    print(f"  Accuracy: {stats['acc']*100:.2f}%")
    print(f"  PPL: {stats['ppl']:.2f}")
    print(f"  Shallow ratio: {stats['shallow_ratio']*100:.1f}%")
    print(f"  Compute cost: {stats['compute_cost']*100:.1f}%")
    print(f"  Compute savings: {(1-stats['compute_cost'])*100:.1f}%")

    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
