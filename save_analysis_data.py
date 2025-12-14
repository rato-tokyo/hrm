"""
Save Analysis Data for Later Analysis

Trains Block 0 once and saves all relevant data to .npz file.
This allows running analysis scripts without re-training.
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
    """Train Block 0 and save all analysis data."""
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
    print("Save Analysis Data")
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

    # Collect all data on validation set
    print("\nCollecting analysis data on validation set...")
    block.eval()

    all_confidences: list[torch.Tensor] = []
    all_actual_probs: list[torch.Tensor] = []
    all_per_token_loss: list[torch.Tensor] = []
    all_exit_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for h, y in val_data.to(str(device)).batches(trainer_config.batch_size, shuffle=False):
            h_out, logits, _ = block.forward(h)

            # Exit classifier confidence
            confidence = block.exit_classifier.compute_confidence(h_out)

            # Actual probability of correct token
            probs = torch.softmax(logits, dim=-1)
            actual_prob = probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

            # Per-token cross-entropy loss
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='none'
            ).view(h_out.size(0), h_out.size(1))

            # Exit labels (what exit_classifier was trained on)
            # BDR-style: exit_labels = loss directly (not exp(-loss))
            exit_labels = per_token_loss

            all_confidences.append(confidence.cpu())
            all_actual_probs.append(actual_prob.cpu())
            all_per_token_loss.append(per_token_loss.cpu())
            all_exit_labels.append(exit_labels.cpu())

    # Flatten all tensors
    confidences = torch.cat(all_confidences).view(-1).numpy()
    actual_probs = torch.cat(all_actual_probs).view(-1).numpy()
    per_token_loss = torch.cat(all_per_token_loss).view(-1).numpy()
    exit_labels = torch.cat(all_exit_labels).view(-1).numpy()

    # Save to .npz file
    output_path = Path("analysis_data.npz")
    np.savez(
        output_path,
        confidences=confidences,
        actual_probs=actual_probs,
        per_token_loss=per_token_loss,
        exit_labels=exit_labels,
        threshold=stats['threshold'],
        best_val_ppl=stats['best_val_ppl'],
    )

    print(f"\nSaved: {output_path}")
    print(f"  Total tokens: {len(confidences)}")
    print(f"  Confidences: mean={confidences.mean():.4f}, std={confidences.std():.4f}")
    print(f"  Exit labels: mean={exit_labels.mean():.4f}, std={exit_labels.std():.4f}")
    print(f"  Actual probs: mean={actual_probs.mean():.4f}, std={actual_probs.std():.4f}")
    print(f"  Per-token loss: mean={per_token_loss.mean():.4f}, std={per_token_loss.std():.4f}")

    # Quick sanity check: Hard/Easy separation
    # BDR-style: predicted_loss > threshold = hard token
    threshold = stats['threshold']
    hard_mask = confidences > threshold  # BDR: high predicted_loss = hard
    easy_mask = ~hard_mask

    print(f"\n{'=' * 60}")
    print("Hard/Easy Separation Check")
    print("=" * 60)
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Hard tokens: {hard_mask.sum()} ({hard_mask.mean()*100:.1f}%)")
    print(f"  Easy tokens: {easy_mask.sum()} ({easy_mask.mean()*100:.1f}%)")
    print()
    print(f"  Hard tokens - Loss: mean={per_token_loss[hard_mask].mean():.4f}, std={per_token_loss[hard_mask].std():.4f}")
    print(f"  Easy tokens - Loss: mean={per_token_loss[easy_mask].mean():.4f}, std={per_token_loss[easy_mask].std():.4f}")
    print()
    print(f"  Hard tokens - Actual prob: mean={actual_probs[hard_mask].mean():.4f}")
    print(f"  Easy tokens - Actual prob: mean={actual_probs[easy_mask].mean():.4f}")

    # Is separation working?
    if per_token_loss[hard_mask].mean() > per_token_loss[easy_mask].mean():
        print("\n  ✓ Separation is WORKING: Hard tokens have higher loss")
    else:
        print("\n  ✗ Separation is NOT working: Hard tokens have lower loss (unexpected)")


if __name__ == "__main__":
    main()
