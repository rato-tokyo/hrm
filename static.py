"""
Exit Classifier Score Distribution Analysis

Trains Block 0 and visualizes the distribution of exit classifier scores
on validation data with Beta Mixture Model and KDE fitting.
"""

import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy.integrate import IntegrationWarning

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


def fit_beta_mixture(
    data: np.ndarray,
    n_components: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Beta Mixture Model using EM algorithm.

    Args:
        data: 1D array of values in (0, 1)
        n_components: Number of beta components
        max_iter: Maximum EM iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (alphas, betas, weights) for each component
    """
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=IntegrationWarning)

    n_samples = len(data)

    # Clip data to avoid numerical issues at boundaries
    data = np.clip(data, 1e-4, 1 - 1e-4)

    # Initialize parameters using k-means style initialization
    sorted_data = np.sort(data)
    split_points = np.linspace(0, n_samples, n_components + 1, dtype=int)
    alphas = np.zeros(n_components)
    betas_arr = np.zeros(n_components)
    weights = np.ones(n_components) / n_components

    for k in range(n_components):
        segment = sorted_data[split_points[k]:split_points[k + 1]]
        if len(segment) > 0:
            mean = np.clip(segment.mean(), 0.01, 0.99)
            var = max(segment.var(), 1e-4)
            # Method of moments for beta distribution
            common = mean * (1 - mean) / var - 1
            common = np.clip(common, 2.0, 50.0)  # Ensure valid parameters
            alphas[k] = max(mean * common, 1.0)
            betas_arr[k] = max((1 - mean) * common, 1.0)

    # EM algorithm
    prev_log_likelihood = -np.inf
    for iteration in range(max_iter):
        # E-step: compute responsibilities
        log_probs = np.zeros((n_samples, n_components))
        for k in range(n_components):
            try:
                log_probs[:, k] = np.log(weights[k] + 1e-10) + scipy_stats.beta.logpdf(
                    data, alphas[k], betas_arr[k]
                )
            except Exception:
                log_probs[:, k] = -1e10

        # Handle invalid values
        log_probs = np.nan_to_num(log_probs, nan=-1e10, posinf=-1e10, neginf=-1e10)

        # Log-sum-exp for numerical stability
        max_log_probs = log_probs.max(axis=1, keepdims=True)
        log_sum = max_log_probs + np.log(np.exp(log_probs - max_log_probs).sum(axis=1, keepdims=True) + 1e-10)
        log_responsibilities = log_probs - log_sum
        responsibilities = np.exp(log_responsibilities)
        responsibilities = np.nan_to_num(responsibilities, nan=1.0/n_components)

        # Compute log-likelihood
        log_likelihood = log_sum.sum()

        # Check convergence
        if abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood

        # M-step: update parameters
        Nk = responsibilities.sum(axis=0) + 1e-10
        weights = Nk / n_samples

        for k in range(n_components):
            resp_k = responsibilities[:, k]
            weighted_data = resp_k * data

            mean_k = np.clip(weighted_data.sum() / Nk[k], 0.01, 0.99)
            var_k = max((resp_k * (data - mean_k)**2).sum() / Nk[k], 1e-4)

            # Method of moments update
            common = mean_k * (1 - mean_k) / var_k - 1
            common = np.clip(common, 2.0, 100.0)
            alphas[k] = max(mean_k * common, 1.0)
            betas_arr[k] = max((1 - mean_k) * common, 1.0)

    return alphas, betas_arr, weights


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

    # Fit Beta Mixture Model
    print("\nFitting Beta Mixture Model (2 components)...")
    alphas, betas, weights = fit_beta_mixture(
        all_scores_flat,
        n_components=2,
        max_iter=100,
        tol=1e-4,
    )

    # Sort by mode (low confidence first)
    modes = (alphas - 1) / (alphas + betas - 2 + 1e-10)
    modes = np.clip(modes, 0, 1)
    sort_idx = np.argsort(modes)
    alphas = alphas[sort_idx]
    betas = betas[sort_idx]
    weights = weights[sort_idx]
    modes = modes[sort_idx]

    # Calculate separation metrics
    # For beta distribution, use mode distance normalized by combined spread
    var1 = (alphas[0] * betas[0]) / ((alphas[0] + betas[0])**2 * (alphas[0] + betas[0] + 1))
    var2 = (alphas[1] * betas[1]) / ((alphas[1] + betas[1])**2 * (alphas[1] + betas[1] + 1))
    separation = abs(modes[1] - modes[0]) / np.sqrt(var1 + var2)

    print("\nBeta Mixture Model Results:")
    print(f"  Component 1 (Hard):  α={alphas[0]:.2f}, β={betas[0]:.2f}, mode={modes[0]:.4f}, weight={weights[0]:.2%}")
    print(f"  Component 2 (Easy):  α={alphas[1]:.2f}, β={betas[1]:.2f}, mode={modes[1]:.4f}, weight={weights[1]:.2%}")
    print(f"  Separation index: {separation:.4f}")
    print(f"  Mode distance: {modes[1] - modes[0]:.4f}")

    # Fit KDE
    print("\nFitting Kernel Density Estimation...")
    kde = scipy_stats.gaussian_kde(all_scores_flat, bw_method='scott')

    # Create histogram with Beta Mixture and KDE overlay
    num_bins = 50
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot range - use actual data range with some padding
    data_min = max(0.001, all_scores_flat.min() - 0.05)
    data_max = min(0.999, all_scores_flat.max() + 0.05)
    x_range = np.linspace(data_min, data_max, 500)

    # Histogram (lowest z-order, drawn first)
    counts, bin_edges, patches = ax.hist(
        all_scores_flat,
        bins=num_bins,
        density=True,
        alpha=0.4,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5,
        label='Observed distribution',
        zorder=1,
    )

    # KDE (z-order 2)
    kde_values = kde(x_range)
    ax.plot(x_range, kde_values, color='navy', linewidth=2, linestyle=':', label='KDE', zorder=2)

    # Combined Beta Mixture (z-order 3)
    beta_combined = np.zeros_like(x_range)
    for i in range(2):
        beta_combined += weights[i] * scipy_stats.beta.pdf(x_range, alphas[i], betas[i])
    ax.plot(x_range, beta_combined, color='crimson', linewidth=2.5, label='Beta Mixture', zorder=3)

    # Individual Beta components (highest z-order, drawn on top)
    colors = ['darkorange', 'forestgreen']
    labels_comp = ['Hard', 'Easy']
    for i in range(2):
        beta_pdf = weights[i] * scipy_stats.beta.pdf(x_range, alphas[i], betas[i])
        ax.plot(
            x_range, beta_pdf,
            color=colors[i],
            linewidth=3,
            linestyle='--',
            label=f"Beta {labels_comp[i]}: α={alphas[i]:.1f}, β={betas[i]:.1f}",
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

    # Add vertical lines at Beta modes
    ax.axvline(x=modes[0], color='darkorange', linestyle='-.', linewidth=1.5, alpha=0.7)
    ax.axvline(x=modes[1], color='forestgreen', linestyle='-.', linewidth=1.5, alpha=0.7)

    # Labels and title
    ax.set_xlabel('Exit Classifier Score (Confidence)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Exit Classifier Score Distribution: Beta Mixture + KDE', fontsize=14)
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

    # Add Beta Mixture stats text box (right side, below legend)
    bmm_stats_text = (
        f"Beta Mixture Analysis\n"
        f"─────────────────\n"
        f"Hard tokens: {weights[0]:.1%}\n"
        f"Easy tokens: {weights[1]:.1%}\n"
        f"Mode distance: {modes[1] - modes[0]:.4f}\n"
        f"Separation: {separation:.4f}"
    )
    ax.text(
        0.98, 0.55, bmm_stats_text,
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
