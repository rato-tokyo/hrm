"""
LEGO Framework Example - Block-wise Training with TRUE Early Exit
"""

from lego import (
    LEGOLLM,
    LEGOBlock,
    TransformerBlock,
    ExperimentConfig,
    TrainerConfig,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
    train_legollm,
    evaluate_legollm,
    create_sequence_data,
)


def main() -> None:
    """Run LEGO block-wise training example."""
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

    # Create model
    blocks = [
        LEGOBlock(
            TransformerBlock(
                config.dim, config.num_heads, num_layers,
                config.ffn_dim, config.max_seq_len, config.causal, config.eps
            )
        )
        for num_layers in config.block_layers
    ]
    model = LEGOLLM(vocab_size, config.dim, blocks).to(device)

    print(f"Layers per block: {[b.num_layers for b in model.blocks]}")

    # Create sequence data from batches
    train_data = create_sequence_data(model, train_batches)
    val_data = create_sequence_data(model, val_batches)
    print(f"Train data: {len(train_data)} sequences ({train_data.num_tokens} tokens)")
    print(f"Val data: {len(val_data)} sequences ({val_data.num_tokens} tokens)")

    # Train all blocks
    train_stats = train_legollm(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=trainer_config,
        lr_decay=0.1,
    )

    # Evaluate
    print(f"\n{'=' * 60}")
    print("Evaluation: TRUE Early Exit")
    print("=" * 60)

    eval_stats = evaluate_legollm(model, val_batches)

    print("\nFinal Results:")
    print(f"  Accuracy: {eval_stats['accuracy']*100:.2f}%")
    print(f"  PPL: {eval_stats['ppl']:.2f}")
    print(f"  Shallow ratio: {eval_stats['shallow_ratio']*100:.1f}%")
    print(f"  Compute cost: {eval_stats['compute_cost']*100:.1f}%")
    print(f"  Compute savings: {eval_stats['compute_savings']*100:.1f}%")

    # Sanity check: final PPL should not exceed worst block val_ppl
    block_stats = train_stats['block_stats']
    worst_block_ppl = max(s['best_val_ppl'] for s in block_stats)
    if eval_stats['ppl'] > worst_block_ppl:
        print("\n" + "!" * 60)
        print("BUG DETECTED: Final PPL exceeds worst block val_ppl!")
        print(f"  Final PPL: {eval_stats['ppl']:.2f}")
        print(f"  Worst block val_ppl: {worst_block_ppl:.2f}")
        print("  This should not happen - please investigate.")
        print("!" * 60)

    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
