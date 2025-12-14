"""
Exit Classifier Score Distribution Analysis

Trains Block 0 and visualizes the distribution of exit classifier scores
on validation data with Gaussian Mixture Model (GMM) fitting.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

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
    """Train Block 0 and visualize exit classifier score distribution."""
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=1024,
        causal=True,
        eps=1e-6,
        seq_len=32,
        num_samples=10000,
        block_layers=(2,),  # Only 1 block
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
    print("Exit Classifier Score Distribution Analysis")
    print("=" * 60)
    print(f"Device: {device}")

    # Setup
    set_seed(42)
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        config.num_samples, trainer_config.batch_size, config.seq_len, seed=42
    )

    # Create model with single block
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

    # Compute exit classifier scores on validation data
    print("\nComputing exit classifier scores on validation data...")
    block.eval()
    all_scores: list[torch.Tensor] = []

    with torch.no_grad():
        for h, _ in val_data.to(str(device)).batches(trainer_config.batch_size, shuffle=False):
            h_out, _, _ = block.forward(h)
            # Get confidence scores
            scores = block.exit_classifier.compute_confidence(h_out)  # (batch, seq_len)
            all_scores.append(scores.cpu())

    all_scores_flat = torch.cat(all_scores).view(-1).numpy()
    print(f"Total tokens: {len(all_scores_flat)}")
    print(f"Score range: [{all_scores_flat.min():.4f}, {all_scores_flat.max():.4f}]")
    print(f"Score mean: {all_scores_flat.mean():.4f}")
    print(f"Score std: {all_scores_flat.std():.4f}")

    # Fit Gaussian Mixture Model (2 components)
    print("\nFitting Gaussian Mixture Model (2 components)...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(all_scores_flat.reshape(-1, 1))

    # Extract GMM parameters
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    # Sort by mean (low confidence first = Hard)
    sort_idx = np.argsort(means)
    means = means[sort_idx]
    stds = stds[sort_idx]
    weights = weights[sort_idx]

    # Calculate separation index: |μ1 - μ2| / sqrt(σ1² + σ2²)
    separation = abs(means[1] - means[0]) / np.sqrt(stds[0]**2 + stds[1]**2)

    print("\nGMM Results:")
    print(f"  Component 1 (Hard):  μ={means[0]:.4f}, σ={stds[0]:.4f}, weight={weights[0]:.2%}")
    print(f"  Component 2 (Easy):  μ={means[1]:.4f}, σ={stds[1]:.4f}, weight={weights[1]:.2%}")
    print(f"  Separation index: {separation:.4f}")
    print(f"  Mean distance: {means[1] - means[0]:.4f}")

    # Create histogram with GMM overlay
    num_bins = 50
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot range
    data_min = max(0.0, all_scores_flat.min() - 0.05)
    data_max = min(1.0, all_scores_flat.max() + 0.05)
    x_range = np.linspace(data_min, data_max, 500)

    # Histogram
    counts, bin_edges, patches = ax.hist(
        all_scores_flat,
        bins=num_bins,
        density=True,
        alpha=0.5,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5,
        label='Observed distribution',
        zorder=1,
    )

    # Combined GMM
    gmm_combined = np.zeros_like(x_range)
    for i in range(2):
        gmm_combined += weights[i] * (1 / (stds[i] * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((x_range - means[i]) / stds[i])**2)
    ax.plot(x_range, gmm_combined, color='crimson', linewidth=2.5, label='GMM', zorder=3)

    # Individual Gaussian components
    colors = ['darkorange', 'forestgreen']
    labels_comp = ['Hard', 'Easy']
    for i in range(2):
        gaussian_pdf = weights[i] * (1 / (stds[i] * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((x_range - means[i]) / stds[i])**2)
        ax.plot(
            x_range, gaussian_pdf,
            color=colors[i],
            linewidth=2.5,
            linestyle='--',
            label=f"{labels_comp[i]}: μ={means[i]:.3f}, σ={stds[i]:.3f}",
            zorder=4 + i,
        )

    # Add threshold line
    ax.axvline(
        x=stats['threshold'],
        color='purple',
        linestyle='-.',
        linewidth=2.5,
        label=f"Threshold = {stats['threshold']:.4f}",
    )

    # Add vertical lines at GMM means
    ax.axvline(x=means[0], color='darkorange', linestyle='-.', linewidth=1.5, alpha=0.7)
    ax.axvline(x=means[1], color='forestgreen', linestyle='-.', linewidth=1.5, alpha=0.7)

    # Labels and title
    ax.set_xlabel('Exit Classifier Score (Confidence)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Exit Classifier Score Distribution with GMM', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(data_min - 0.02, data_max + 0.02)

    # Add statistics text box (left side)
    basic_stats_text = (
        f"N = {len(all_scores_flat):,}\n"
        f"Mean = {all_scores_flat.mean():.4f}\n"
        f"Std = {all_scores_flat.std():.4f}\n"
        f"Min = {all_scores_flat.min():.4f}\n"
        f"Max = {all_scores_flat.max():.4f}"
    )
    ax.text(
        0.02, 0.98, basic_stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    # Add GMM stats text box (right side, below legend)
    gmm_stats_text = (
        f"GMM Analysis\n"
        f"─────────────────\n"
        f"Hard tokens: {weights[0]:.1%}\n"
        f"Easy tokens: {weights[1]:.1%}\n"
        f"Mean distance: {means[1] - means[0]:.4f}\n"
        f"Separation: {separation:.4f}"
    )
    ax.text(
        0.98, 0.55, gmm_stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig('exit_classifier_distribution.png', dpi=150)
    print("\nSaved plot to: exit_classifier_distribution.png")
    plt.show()


if __name__ == "__main__":
    main()
