"""
Save Hidden States for Loss Prediction Analysis

Trains Block 0 and saves hidden_states along with per_token_loss.
This allows training various models (MLP, etc.) to predict loss from hidden states.
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


def main() -> None:
    """Train Block 0 and save hidden_states with loss."""
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=1024,
        causal=True,
        eps=1e-6,
        seq_len=32,
        num_samples=10000,
        block_layers=(2,),
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
    print("Save Hidden States for Loss Prediction")
    print("=" * 60)
    print(f"Device: {device}")

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

    print(f"\nBlock 0 Results:")
    print(f"  Best PPL: {stats['best_val_ppl']:.2f}")
    print(f"  Threshold: {stats['threshold']:.4f}")

    # Collect hidden_states and loss on validation set
    print("\nCollecting hidden_states and loss on validation set...")
    block.eval()

    all_hidden_states: list[torch.Tensor] = []
    all_per_token_loss: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

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
            all_targets.append(y.cpu())

    # Concatenate and flatten
    hidden_states = torch.cat(all_hidden_states)  # (num_sequences, seq_len, dim)
    per_token_loss = torch.cat(all_per_token_loss)  # (num_sequences, seq_len)
    targets = torch.cat(all_targets)  # (num_sequences, seq_len)

    # Flatten to token level
    num_sequences, seq_len, dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, dim).numpy()  # (num_tokens, dim)
    per_token_loss_flat = per_token_loss.view(-1).numpy()  # (num_tokens,)
    targets_flat = targets.view(-1).numpy()  # (num_tokens,)

    # Estimate file size
    size_mb = (hidden_states_flat.nbytes + per_token_loss_flat.nbytes + targets_flat.nbytes) / (1024 * 1024)
    print(f"\nData size estimate: {size_mb:.1f} MB")

    # Save to .npz file
    output_path = Path("hidden_states_data.npz")
    np.savez(
        output_path,
        hidden_states=hidden_states_flat,
        per_token_loss=per_token_loss_flat,
        targets=targets_flat,
        dim=dim,
        seq_len=seq_len,
        num_sequences=num_sequences,
        best_val_ppl=stats['best_val_ppl'],
        threshold=stats['threshold'],
    )

    print(f"\nSaved: {output_path}")
    print(f"  Hidden states: {hidden_states_flat.shape} ({hidden_states_flat.dtype})")
    print(f"  Per-token loss: {per_token_loss_flat.shape} ({per_token_loss_flat.dtype})")
    print(f"  Targets: {targets_flat.shape} ({targets_flat.dtype})")
    print(f"  Total tokens: {len(per_token_loss_flat):,}")
    print(f"  Dim: {dim}")

    # Statistics
    print(f"\n{'=' * 60}")
    print("Data Statistics")
    print("=" * 60)
    print(f"  Hidden states: mean={hidden_states_flat.mean():.4f}, std={hidden_states_flat.std():.4f}")
    print(f"  Per-token loss: mean={per_token_loss_flat.mean():.4f}, std={per_token_loss_flat.std():.4f}")
    print(f"  Loss range: [{per_token_loss_flat.min():.4f}, {per_token_loss_flat.max():.4f}]")

    # Loss distribution
    print(f"\nLoss quantiles:")
    for q in [0, 25, 50, 75, 100]:
        val = np.percentile(per_token_loss_flat, q)
        print(f"  {q:3d}%: {val:.4f}")


if __name__ == "__main__":
    main()
