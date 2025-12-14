"""
Save Hidden States for Loss Prediction Analysis

Matches the exact configuration used in save_analysis_data.py:
- dim=64, num_heads=4, num_layers=2
- seq_len=32, num_samples=10000
- WikiText-2 dataset

This saves hidden_states along with per_token_loss for MLP training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from lego import (
    LEGOLLM,
    LEGOBlock,
    TransformerBlock,
    ExperimentConfig,
    TrainerConfig,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
    train_block,
    create_sequence_data,
)


def main():
    # EXACT same configuration as save_analysis_data.py
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=1024,
        causal=True,
        eps=1e-6,
        seq_len=32,
        num_samples=10000,
        block_layers=(2,),  # 2 layers only
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
    print("Save Hidden States (LEGO - Same Config as analysis_data.npz)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: dim={config.dim}, heads={config.num_heads}, layers={config.block_layers}")

    # Setup (same seed as save_analysis_data.py)
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
            ),
        )
        for num_layers in config.block_layers
    ]
    model = LEGOLLM(vocab_size, config.dim, blocks).to(device)

    # Create sequence data
    train_data = create_sequence_data(model, train_batches)
    val_data = create_sequence_data(model, val_batches)
    print(f"Train data: {len(train_data)} sequences ({train_data.num_tokens} tokens)")
    print(f"Val data: {len(val_data)} sequences ({val_data.num_tokens} tokens)")

    # Train Block 0
    block = model.blocks[0]
    optimizer = torch.optim.AdamW(block.parameters(), lr=trainer_config.lr)

    hard_data, stats = train_block(
        block=block,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        config=trainer_config,
    )

    print(f"\nBlock 0 Training Results:")
    print(f"  Total epochs: {stats['total_epochs']}")
    print(f"  Best epoch: {stats['best_epoch'] + 1}")
    print(f"  Early stopped: {stats['stopped_early']}")
    print(f"  Best val PPL: {stats['best_val_ppl']:.2f}")
    print(f"  Threshold: {stats['threshold']:.4f}")

    # LLM sanity check
    print(f"\n{'=' * 60}")
    print("LLM Sanity Check")
    print("=" * 60)

    # Check PPL improved from random (vocab_size ~ 50000, random PPL ~ 50000)
    random_ppl = vocab_size
    if stats['best_val_ppl'] < random_ppl * 0.1:
        print(f"  ✓ PPL ({stats['best_val_ppl']:.0f}) << random ({random_ppl}) - Model is learning")
    else:
        print(f"  ✗ PPL ({stats['best_val_ppl']:.0f}) too high - Model may not be learning")

    # Check early stopping worked
    if stats['stopped_early']:
        print(f"  ✓ Early stopping triggered at epoch {stats['total_epochs']}")
    else:
        print(f"  △ No early stopping (ran all {stats['total_epochs']} epochs)")

    # Check train/val PPL trend
    train_ppls = stats['train_ppls']
    val_ppls = stats['val_ppls']
    if len(train_ppls) >= 2:
        if train_ppls[-1] < train_ppls[0]:
            print(f"  ✓ Train PPL decreased: {train_ppls[0]:.0f} → {train_ppls[-1]:.0f}")
        else:
            print(f"  ✗ Train PPL did not decrease: {train_ppls[0]:.0f} → {train_ppls[-1]:.0f}")

    # Show PPL history
    print(f"\n  PPL History (first 5, last 3):")
    for i, (tr, va) in enumerate(zip(train_ppls[:5], val_ppls[:5])):
        print(f"    Epoch {i+1}: train={tr:.0f}, val={va:.0f}")
    if len(train_ppls) > 5:
        print(f"    ...")
        for i in range(max(5, len(train_ppls)-3), len(train_ppls)):
            print(f"    Epoch {i+1}: train={train_ppls[i]:.0f}, val={val_ppls[i]:.0f}")

    # Collect hidden_states and loss on validation set
    print("\nCollecting hidden_states on validation set...")
    block.eval()

    all_hidden_states: list[torch.Tensor] = []
    all_per_token_loss: list[torch.Tensor] = []

    with torch.no_grad():
        for h, y in val_data.to(str(device)).batches(trainer_config.batch_size, shuffle=False):
            h_out, logits, _ = block.forward(h)

            # Per-token cross-entropy loss
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='none'
            ).view(h_out.size(0), h_out.size(1))

            all_hidden_states.append(h_out.cpu())
            all_per_token_loss.append(per_token_loss.cpu())

    # Flatten to token level
    hidden_states = torch.cat(all_hidden_states)  # (num_sequences, seq_len, dim)
    per_token_loss = torch.cat(all_per_token_loss)  # (num_sequences, seq_len)

    num_sequences, seq_len, dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, dim).numpy()  # (num_tokens, dim)
    per_token_loss_flat = per_token_loss.view(-1).numpy()  # (num_tokens,)

    # Statistics
    print(f"\n{'=' * 60}")
    print("Data Statistics")
    print("=" * 60)
    print(f"  Hidden states: {hidden_states_flat.shape} (tokens x dim)")
    print(f"  Per-token loss: {per_token_loss_flat.shape}")
    print(f"  Total tokens: {len(per_token_loss_flat):,}")
    print(f"  Loss: mean={per_token_loss_flat.mean():.4f}, std={per_token_loss_flat.std():.4f}")
    print(f"  Loss range: [{per_token_loss_flat.min():.4f}, {per_token_loss_flat.max():.4f}]")

    # Loss quantiles
    print(f"\nLoss quantiles:")
    for q in [0, 25, 50, 75, 100]:
        val = np.percentile(per_token_loss_flat, q)
        print(f"  {q:3d}%: {val:.4f}")

    # Save
    output_path = Path("hidden_states_data.npz")
    size_mb = (hidden_states_flat.nbytes + per_token_loss_flat.nbytes) / (1024 * 1024)
    print(f"\nSaving to {output_path} ({size_mb:.1f} MB)...")

    np.savez(
        output_path,
        hidden_states=hidden_states_flat.astype(np.float32),
        per_token_loss=per_token_loss_flat.astype(np.float32),
        dim=dim,
        seq_len=seq_len,
        num_sequences=num_sequences,
        best_val_ppl=stats['best_val_ppl'],
        threshold=stats['threshold'],
    )

    print(f"\nSaved: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
