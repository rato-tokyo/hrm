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
    Fit a Generalized Gaussian Mixture Model using EM algorithm.

    Args:
        data: 1D array of values
        n_components: Number of GGD components
        max_iter: Maximum EM iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (betas, locs, scales, weights) for each component
    """
    n_samples = len(data)

    # Initialize parameters using k-means style initialization
    sorted_data = np.sort(data)
    split_points = np.linspace(0, n_samples, n_components + 1, dtype=int)

    betas = np.ones(n_components) * 2.0  # Start with Gaussian
    locs = np.zeros(n_components)
    scales = np.ones(n_components)
    weights = np.ones(n_components) / n_components

    for k in range(n_components):
        segment = sorted_data[split_points[k]:split_points[k + 1]]
        if len(segment) > 0:
            locs[k] = segment.mean()
            scales[k] = max(segment.std(), 0.01)

    # EM algorithm
    prev_log_likelihood = -np.inf

    for iteration in range(max_iter):
        # E-step: compute responsibilities
        log_probs = np.zeros((n_samples, n_components))
        for k in range(n_components):
            try:
                log_probs[:, k] = np.log(weights[k] + 1e-10) + gennorm.logpdf(
                    data, betas[k], loc=locs[k], scale=scales[k]
                )
            except Exception:
                log_probs[:, k] = -1e10

        # Handle invalid values
        log_probs = np.nan_to_num(log_probs, nan=-1e10, posinf=-1e10, neginf=-1e10)

        # Log-sum-exp for numerical stability
        max_log_probs = log_probs.max(axis=1, keepdims=True)
        log_sum = max_log_probs + np.log(
            np.exp(log_probs - max_log_probs).sum(axis=1, keepdims=True) + 1e-10
        )
        log_responsibilities = log_probs - log_sum
        responsibilities = np.exp(log_responsibilities)
        responsibilities = np.nan_to_num(responsibilities, nan=1.0 / n_components)

        # Compute log-likelihood
        log_likelihood = log_sum.sum()

        # Check convergence
        if abs(log_likelihood - prev_log_likelihood) < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break
        prev_log_likelihood = log_likelihood

        # M-step: update parameters
        Nk = responsibilities.sum(axis=0) + 1e-10
        weights = Nk / n_samples

        for k in range(n_components):
            resp_k = responsibilities[:, k]
            weighted_data = resp_k * data

            # Update location (weighted mean)
            locs[k] = weighted_data.sum() / Nk[k]

            # Optimize beta and scale for this component
            def neg_log_likelihood(params: np.ndarray) -> float:
                beta, scale = params
                if beta <= 0.1 or scale <= 0.001:
                    return 1e10
                try:
                    ll = (resp_k * gennorm.logpdf(data, beta, loc=locs[k], scale=scale)).sum()
                    return -ll if np.isfinite(ll) else 1e10
                except Exception:
                    return 1e10

            # Optimize
            result = minimize(
                neg_log_likelihood,
                x0=[betas[k], scales[k]],
                method='L-BFGS-B',
                bounds=[(0.2, 10.0), (0.001, 2.0)],
            )
            if result.success:
                betas[k], scales[k] = result.x

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
