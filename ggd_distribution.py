"""
Generalized Gaussian Distribution (GGD) Visualization

Compares different β values and fits GGD Mixture Model to exit classifier scores.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gennorm
from scipy.optimize import minimize

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


def fit_ggd_mixture(
    data: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Generalized Gaussian Mixture Model using hard assignment + grid search.

    This approach:
    1. Uses threshold-based hard assignment (not soft EM)
    2. Fits each component independently with grid search over beta

    Args:
        data: 1D array of values
        n_components: Number of GGD components
        max_iter: Maximum iterations (unused, kept for API compatibility)
        tol: Convergence tolerance (unused, kept for API compatibility)

    Returns:
        Tuple of (betas, locs, scales, weights) for each component
    """
    # Hard assignment: split at median
    threshold = np.median(data)
    hard_mask = data < threshold
    easy_mask = ~hard_mask

    hard_data = data[hard_mask]
    easy_data = data[easy_mask]

    print(f"  Hard assignment: {len(hard_data)} hard, {len(easy_data)} easy tokens")

    def fit_single_ggd(subset: np.ndarray) -> tuple[float, float, float]:
        """Fit single GGD to data subset using grid search."""
        loc = np.median(subset)  # Use median for robustness

        best_beta = 2.0
        best_scale = subset.std()
        best_ll = -np.inf

        # Grid search over beta
        for beta in np.linspace(0.3, 4.0, 20):
            # Estimate scale using MLE formula for GGD
            # For GGD, scale relates to: E[|x-μ|^β] = σ^β * Γ(2/β) / Γ(1/β)
            centered = np.abs(subset - loc)

            # Try multiple scale values
            for scale in np.linspace(0.02, 0.3, 15):
                try:
                    ll = gennorm.logpdf(subset, beta, loc=loc, scale=scale).sum()
                    if np.isfinite(ll) and ll > best_ll:
                        best_ll = ll
                        best_beta = beta
                        best_scale = scale
                except Exception:
                    continue

        return best_beta, loc, best_scale

    # Fit each component
    beta_hard, loc_hard, scale_hard = fit_single_ggd(hard_data)
    beta_easy, loc_easy, scale_easy = fit_single_ggd(easy_data)

    print(f"  Hard fit: β={beta_hard:.2f}, μ={loc_hard:.4f}, σ={scale_hard:.4f}")
    print(f"  Easy fit: β={beta_easy:.2f}, μ={loc_easy:.4f}, σ={scale_easy:.4f}")

    betas = np.array([beta_hard, beta_easy])
    locs = np.array([loc_hard, loc_easy])
    scales = np.array([scale_hard, scale_easy])
    weights = np.array([len(hard_data) / len(data), len(easy_data) / len(data)])

    return betas, locs, scales, weights


def plot_ggd_comparison() -> None:
    """Plot GGD with different β values for comparison."""
    x = np.linspace(-4, 4, 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    beta_values = [
        (0.5, 'β=0.5 (Super-peaked)', 'red'),
        (1.0, 'β=1.0 (Laplace)', 'orange'),
        (2.0, 'β=2.0 (Gaussian)', 'blue'),
        (5.0, 'β=5.0 (Flat-ish)', 'green'),
    ]

    for beta, label, color in beta_values:
        pdf = gennorm.pdf(x, beta)
        ax.plot(x, pdf, label=label, linewidth=2.5, color=color)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Generalized Gaussian Distribution: Effect of β', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)

    plt.tight_layout()
    plt.savefig('ggd_comparison.png', dpi=150)
    print("Saved: ggd_comparison.png")
    plt.show()


def main() -> None:
    """Train Block 0 and fit GGD Mixture Model to exit classifier scores."""

    # First, show GGD comparison
    print("=" * 60)
    print("Part 1: GGD Shape Comparison")
    print("=" * 60)
    plot_ggd_comparison()

    # Now train model and analyze scores
    print("\n" + "=" * 60)
    print("Part 2: GGD Mixture Model on Exit Classifier Scores")
    print("=" * 60)

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
        exit_hidden_dim=128,
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
            config.exit_hidden_dim,
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

    # Compute exit classifier scores
    print("\nComputing exit classifier scores on validation data...")
    block.eval()
    all_scores: list[torch.Tensor] = []

    with torch.no_grad():
        for h, _ in val_data.to(str(device)).batches(trainer_config.batch_size, shuffle=False):
            h_out, _, _ = block.forward(h)
            scores = block.exit_classifier.compute_confidence(h_out)
            all_scores.append(scores.cpu())

    all_scores_flat = torch.cat(all_scores).view(-1).numpy()
    print(f"Total tokens: {len(all_scores_flat)}")
    print(f"Score range: [{all_scores_flat.min():.4f}, {all_scores_flat.max():.4f}]")
    print(f"Score mean: {all_scores_flat.mean():.4f}")
    print(f"Score std: {all_scores_flat.std():.4f}")

    # Fit GGD Mixture Model
    print("\nFitting GGD Mixture Model (2 components)...")
    betas, locs, scales, weights = fit_ggd_mixture(
        all_scores_flat,
        n_components=2,
        max_iter=50,
        tol=1e-4,
    )

    # Sort by location (low confidence first = Hard)
    sort_idx = np.argsort(locs)
    betas = betas[sort_idx]
    locs = locs[sort_idx]
    scales = scales[sort_idx]
    weights = weights[sort_idx]

    # Interpret β values
    def interpret_beta(beta: float) -> str:
        if beta < 1.0:
            return "super-peaked"
        elif beta < 1.5:
            return "Laplace-like"
        elif beta < 2.5:
            return "Gaussian-like"
        else:
            return "flat-ish"

    print("\nGGD Mixture Model Results:")
    print(f"  Component 1 (Hard):  β={betas[0]:.2f} ({interpret_beta(betas[0])}), "
          f"μ={locs[0]:.4f}, σ={scales[0]:.4f}, weight={weights[0]:.2%}")
    print(f"  Component 2 (Easy):  β={betas[1]:.2f} ({interpret_beta(betas[1])}), "
          f"μ={locs[1]:.4f}, σ={scales[1]:.4f}, weight={weights[1]:.2%}")

    # Create visualization
    num_bins = 50
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot range
    data_min = max(0.0, all_scores_flat.min() - 0.05)
    data_max = min(1.0, all_scores_flat.max() + 0.05)
    x_range = np.linspace(data_min, data_max, 500)

    # Histogram
    ax.hist(
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

    # Combined GGD Mixture
    ggd_combined = np.zeros_like(x_range)
    for i in range(2):
        ggd_combined += weights[i] * gennorm.pdf(x_range, betas[i], loc=locs[i], scale=scales[i])
    ax.plot(x_range, ggd_combined, color='crimson', linewidth=2.5, label='GGD Mixture', zorder=3)

    # Individual GGD components
    colors = ['darkorange', 'forestgreen']
    labels_comp = ['Hard', 'Easy']
    for i in range(2):
        ggd_pdf = weights[i] * gennorm.pdf(x_range, betas[i], loc=locs[i], scale=scales[i])
        ax.plot(
            x_range, ggd_pdf,
            color=colors[i],
            linewidth=2.5,
            linestyle='--',
            label=f"{labels_comp[i]}: β={betas[i]:.2f}, μ={locs[i]:.3f}",
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

    # Add vertical lines at GGD means
    ax.axvline(x=locs[0], color='darkorange', linestyle='-.', linewidth=1.5, alpha=0.7)
    ax.axvline(x=locs[1], color='forestgreen', linestyle='-.', linewidth=1.5, alpha=0.7)

    # Labels and title
    ax.set_xlabel('Exit Classifier Score (Confidence)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Exit Classifier Score Distribution with GGD Mixture', fontsize=14)
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

    # Add GGD Mixture stats text box
    ggd_stats_text = (
        f"GGD Mixture Analysis\n"
        f"─────────────────\n"
        f"Hard: β={betas[0]:.2f} ({interpret_beta(betas[0])})\n"
        f"Easy: β={betas[1]:.2f} ({interpret_beta(betas[1])})\n"
        f"Hard tokens: {weights[0]:.1%}\n"
        f"Easy tokens: {weights[1]:.1%}"
    )
    ax.text(
        0.98, 0.55, ggd_stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig('ggd_mixture_distribution.png', dpi=150)
    print("\nSaved: ggd_mixture_distribution.png")
    plt.show()

    # Compare β values with interpretation
    print("\n" + "=" * 60)
    print("Interpretation of β values")
    print("=" * 60)
    print(f"  β < 1.0  : Super-peaked (sharper than Laplace)")
    print(f"  β = 1.0  : Laplace distribution")
    print(f"  β = 2.0  : Gaussian distribution")
    print(f"  β > 2.0  : Flatter than Gaussian")
    print()
    print(f"  Hard component β = {betas[0]:.2f} → {interpret_beta(betas[0])}")
    print(f"  Easy component β = {betas[1]:.2f} → {interpret_beta(betas[1])}")


if __name__ == "__main__":
    main()
