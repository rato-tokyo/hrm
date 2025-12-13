"""
LEGO Framework Example - Block-wise Training with TRUE Early Exit

Workflow:
1. Create LEGOLLM with multiple blocks
2. Train Block 0 on all data, collect hard examples
3. Train Block 1 on hard examples only
4. Evaluate with TRUE Early Exit
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
from lego import (
    LEGOLLM,
    LEGOBlock,
    TrainingData,
    ExperimentConfig,
    train_block,
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
    # Thresholds are set automatically by train_block() based on hard_ratio
    blocks = [
        LEGOBlock(config.dim, config.num_heads, config.phase1_layers),
        LEGOBlock(config.dim, config.num_heads, config.phase2_layers - config.phase1_layers),
    ]
    model = LEGOLLM(vocab_size, config.dim, blocks).to(device)

    print(f"Blocks: {len(model.blocks)}")
    print(f"Layers per block: {[b.num_layers for b in model.blocks]}")

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
    hard_data, stats0 = train_block(
        block=model.blocks[0],
        data=initial_data,
        optimizer=optimizer0,
        batch_size=config.phase1_batch,
        max_epochs=config.phase1_epochs,
        patience=config.phase1_patience,
        hard_ratio=config.hard_example_ratio,
        verbose=True
    )

    print(f"\nBlock 0 Results:")
    print(f"  Best PPL: {stats0['best_val_ppl']:.2f}")
    print(f"  Threshold: {stats0['threshold']:.4f}")
    print(f"  Hard examples: {len(hard_data)} ({stats0['hard_ratio']*100:.1f}%)")

    # Phase 2: Train Block 1 on hard examples only
    print(f"\n{'=' * 60}")
    print("Phase 2: Train Block 1 on hard examples")
    print("=" * 60)

    if len(hard_data) > 0:
        optimizer1 = torch.optim.AdamW(model.blocks[1].parameters(), lr=config.phase2_lr)
        _, stats1 = train_block(
            block=model.blocks[1],
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

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    all_exit_counts = [0] * len(model.blocks)

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            logits, stats = model.forward(x, return_stats=True)

            # Accumulate exit counts
            for i, count in enumerate(stats['exit_counts']):
                all_exit_counts[i] += count

            # Loss and accuracy
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            total_loss += F.cross_entropy(logits_flat, y_flat, reduction='sum').item()
            total_tokens += y_flat.numel()
            correct += (logits_flat.argmax(dim=-1) == y_flat).sum().item()

    import numpy as np
    ppl = float(np.exp(total_loss / total_tokens))
    acc = correct / total_tokens
    shallow_exits = sum(all_exit_counts[:-1])
    shallow_ratio = shallow_exits / total_tokens if total_tokens > 0 else 0.0

    # Compute cost
    total_layers_computed = 0
    layers_so_far = 0
    for block_idx, count in enumerate(all_exit_counts):
        layers_so_far += model.blocks[block_idx].num_layers
        total_layers_computed += count * layers_so_far
    compute_cost = total_layers_computed / (total_tokens * model.num_layers) if total_tokens > 0 else 1.0

    print(f"\nFinal Results:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  PPL: {ppl:.2f}")
    print(f"  Shallow ratio: {shallow_ratio*100:.1f}%")
    print(f"  Compute cost: {compute_cost*100:.1f}%")
    print(f"  Compute savings: {(1-compute_cost)*100:.1f}%")

    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
