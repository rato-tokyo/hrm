"""
LEGO Framework Example - Block-wise Training with TRUE Early Exit

Workflow:
1. Create LEGOLLM with multiple blocks
2. Train Block 0 on all data, collect hard examples (sequences with low confidence tokens)
3. Train Block 1 on hard examples only
4. Evaluate with TRUE Early Exit
"""

import torch
import torch.nn.functional as F
from lego import (
    LEGOLLM,
    LEGOBlock,
    TransformerBlock,
    SequenceData,
    ExperimentConfig,
    TrainerConfig,
    train_block,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
)


def main() -> None:
    """Run LEGO block-wise training example."""
    # Configuration - all values must be explicitly specified
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=1024,
        causal=True,
        eps=1e-6,
        seq_len=32,
        num_samples=10000,
        block_layers=(2, 2),
    )

    trainer_config = TrainerConfig(
        batch_size=64,
        max_epochs=50,
        patience=3,
        grad_clip=1.0,
        val_ratio=0.2,
        hard_ratio=0.5,
        lr=1e-3,
        verbose=True,
    )

    device = get_device()

    print("=" * 60)
    print("LEGO Framework - Block-wise Training Example")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: dim={config.dim}, heads={config.num_heads}")
    print(f"Blocks: {config.block_layers}")

    # Setup
    set_seed(42)
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        config.num_samples, trainer_config.batch_size, config.seq_len, seed=42
    )

    # Create model with blocks based on config.block_layers
    blocks = [
        LEGOBlock(
            TransformerBlock(
                config.dim, config.num_heads, num_layers,
                config.ffn_dim, config.max_seq_len, config.causal, config.eps
            ),
        )
        for num_layers in config.block_layers
    ]
    model = LEGOLLM(vocab_size, config.dim, blocks).to(device)

    print(f"Layers per block: {[b.num_layers for b in model.blocks]}")

    # Phase 1: Train Block 0 on all data
    print(f"\n{'=' * 60}")
    print("Phase 1: Train Block 0 on all data")
    print("=" * 60)

    # Create initial SequenceData from train batches
    # Embed tokens to get hidden states: (num_sequences, seq_len, dim)
    all_hidden = []
    all_targets = []
    with torch.no_grad():
        for x, y in train_batches:
            x, y = x.to(device), y.to(device)
            h = model.embedding(x)  # (batch, seq_len, dim)
            all_hidden.append(h)
            all_targets.append(y)

    initial_data = SequenceData(
        torch.cat(all_hidden),  # (num_sequences, seq_len, dim)
        torch.cat(all_targets)  # (num_sequences, seq_len)
    )
    print(f"Initial data: {len(initial_data)} sequences ({initial_data.num_tokens} tokens)")

    # Train Block 0
    optimizer0 = torch.optim.AdamW(model.blocks[0].parameters(), lr=trainer_config.lr)
    hard_data, stats0 = train_block(
        block=model.blocks[0],
        data=initial_data,
        optimizer=optimizer0,
        config=trainer_config,
    )

    print("\nBlock 0 Results:")
    print(f"  Best PPL: {stats0['best_val_ppl']:.2f}")
    print(f"  Threshold: {stats0['threshold']:.4f}")
    print(f"  Hard examples: {len(hard_data)} sequences ({hard_data.num_tokens} tokens, {stats0['hard_ratio']*100:.1f}%)")

    # Phase 2: Train Block 1 on hard examples only
    # hard_data contains Block 0's output hidden states and targets
    # Ready to use directly as Block 1's input
    print(f"\n{'=' * 60}")
    print("Phase 2: Train Block 1 on hard examples")
    print("=" * 60)

    if len(hard_data) > 0:
        # Use lower learning rate for deeper block
        phase2_config = TrainerConfig(
            batch_size=trainer_config.batch_size,
            max_epochs=trainer_config.max_epochs,
            patience=trainer_config.patience,
            grad_clip=trainer_config.grad_clip,
            val_ratio=trainer_config.val_ratio,
            hard_ratio=trainer_config.hard_ratio,
            lr=trainer_config.lr * 0.1,
            verbose=trainer_config.verbose,
        )
        optimizer1 = torch.optim.AdamW(model.blocks[1].parameters(), lr=phase2_config.lr)
        _, stats1 = train_block(
            block=model.blocks[1],
            data=hard_data,
            optimizer=optimizer1,
            config=phase2_config,
        )
        print("\nBlock 1 Results:")
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
    correct: int = 0
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
            correct += int((logits_flat.argmax(dim=-1) == y_flat).sum().item())

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

    print("\nFinal Results:")
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
