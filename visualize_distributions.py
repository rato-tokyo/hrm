"""
Visualize BDR-style Exit Classifier Distributions

Creates comprehensive visualizations of:
1. Predicted loss distribution (exit_classifier output)
2. Actual loss distribution
3. Correlation between predicted and actual loss
4. Hard/Easy separation quality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from pathlib import Path


def load_data(path: str = "analysis_data.npz") -> dict:
    """Load saved analysis data."""
    data = np.load(path)
    return {
        'predicted_loss': data['confidences'],  # BDR: confidence = predicted_loss
        'actual_loss': data['per_token_loss'],
        'actual_probs': data['actual_probs'],
        'threshold': float(data['threshold']),
        'best_val_ppl': float(data['best_val_ppl']),
    }


def plot_main_distributions(data: dict) -> None:
    """Plot main distributions: predicted vs actual loss."""
    predicted_loss = data['predicted_loss']
    actual_loss = data['actual_loss']
    threshold = data['threshold']

    # BDR-style: high predicted_loss > threshold = hard
    hard_mask = predicted_loss > threshold
    easy_mask = ~hard_mask

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Predicted loss distribution (exit_classifier output)
    ax1 = axes[0, 0]
    ax1.hist(predicted_loss, bins=100, alpha=0.7, color='blue', density=True)
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold={threshold:.2f}')
    ax1.axvline(predicted_loss.mean(), color='orange', linestyle='-', linewidth=2,
                label=f'Mean={predicted_loss.mean():.2f}')
    ax1.set_xlabel('Predicted Loss')
    ax1.set_ylabel('Density')
    ax1.set_title('Exit Classifier Output Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Actual loss distribution
    ax2 = axes[0, 1]
    ax2.hist(actual_loss, bins=100, alpha=0.7, color='green', density=True)
    ax2.axvline(actual_loss.mean(), color='orange', linestyle='-', linewidth=2,
                label=f'Mean={actual_loss.mean():.2f}')
    ax2.set_xlabel('Actual Loss (Cross-Entropy)')
    ax2.set_ylabel('Density')
    ax2.set_title('Actual Loss Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Predicted vs Actual scatter
    ax3 = axes[0, 2]
    n_sample = min(5000, len(predicted_loss))
    idx = np.random.choice(len(predicted_loss), n_sample, replace=False)
    ax3.scatter(actual_loss[idx], predicted_loss[idx], alpha=0.2, s=3)

    # Add diagonal line (perfect prediction)
    min_val = min(actual_loss.min(), predicted_loss.min())
    max_val = max(actual_loss.max(), predicted_loss.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (perfect)')

    # Add regression line
    slope, intercept, r, p, se = scipy_stats.linregress(actual_loss, predicted_loss)
    x_line = np.linspace(actual_loss.min(), actual_loss.max(), 100)
    ax3.plot(x_line, slope * x_line + intercept, 'g-', linewidth=2,
             label=f'Regression (r={r:.3f})')

    ax3.set_xlabel('Actual Loss')
    ax3.set_ylabel('Predicted Loss')
    ax3.set_title('Predicted vs Actual Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Hard vs Easy - Predicted Loss
    ax4 = axes[1, 0]
    ax4.hist(predicted_loss[hard_mask], bins=50, alpha=0.6, color='red',
             density=True, label=f'Hard (n={hard_mask.sum():,})')
    ax4.hist(predicted_loss[easy_mask], bins=50, alpha=0.6, color='green',
             density=True, label=f'Easy (n={easy_mask.sum():,})')
    ax4.axvline(threshold, color='purple', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Loss')
    ax4.set_ylabel('Density')
    ax4.set_title('Predicted Loss: Hard vs Easy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Hard vs Easy - Actual Loss
    ax5 = axes[1, 1]
    ax5.hist(actual_loss[hard_mask], bins=50, alpha=0.6, color='red',
             density=True, label=f'Hard (mean={actual_loss[hard_mask].mean():.2f})')
    ax5.hist(actual_loss[easy_mask], bins=50, alpha=0.6, color='green',
             density=True, label=f'Easy (mean={actual_loss[easy_mask].mean():.2f})')
    ax5.set_xlabel('Actual Loss')
    ax5.set_ylabel('Density')
    ax5.set_title('Actual Loss: Hard vs Easy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Boxplot comparison
    ax6 = axes[1, 2]
    box_data = [
        actual_loss[hard_mask],
        actual_loss[easy_mask],
    ]
    bp = ax6.boxplot(box_data, labels=['Hard\n(predicted)', 'Easy\n(predicted)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('green')
    bp['boxes'][1].set_alpha(0.5)
    ax6.set_ylabel('Actual Loss')
    ax6.set_title('Actual Loss by Predicted Category')
    ax6.grid(True, alpha=0.3)

    # Add means as horizontal lines
    for i, (box, mask) in enumerate(zip(bp['boxes'], [hard_mask, easy_mask])):
        mean_val = actual_loss[mask].mean()
        ax6.hlines(mean_val, i + 0.75, i + 1.25, colors='black', linewidth=2)

    plt.tight_layout()
    plt.savefig('bdr_distributions.png', dpi=150)
    print("Saved: bdr_distributions.png")
    plt.close()


def plot_correlation_analysis(data: dict) -> None:
    """Plot detailed correlation analysis."""
    predicted_loss = data['predicted_loss']
    actual_loss = data['actual_loss']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 2D Histogram (heatmap style)
    ax1 = axes[0]
    h = ax1.hist2d(actual_loss, predicted_loss, bins=50, cmap='YlOrRd')
    plt.colorbar(h[3], ax=ax1, label='Count')

    # Add diagonal
    min_val = min(actual_loss.min(), predicted_loss.min())
    max_val = max(actual_loss.max(), predicted_loss.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='y=x')

    ax1.set_xlabel('Actual Loss')
    ax1.set_ylabel('Predicted Loss')
    ax1.set_title('Predicted vs Actual Loss (2D Histogram)')
    ax1.legend()

    # 2. Binned analysis
    ax2 = axes[1]

    # Bin by actual loss
    n_bins = 20
    bin_edges = np.percentile(actual_loss, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_pred_means = []
    bin_pred_stds = []

    for i in range(n_bins):
        mask = (actual_loss >= bin_edges[i]) & (actual_loss < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_pred_means.append(predicted_loss[mask].mean())
            bin_pred_stds.append(predicted_loss[mask].std())

    bin_centers = np.array(bin_centers)
    bin_pred_means = np.array(bin_pred_means)
    bin_pred_stds = np.array(bin_pred_stds)

    ax2.errorbar(bin_centers, bin_pred_means, yerr=bin_pred_stds,
                 fmt='o-', capsize=3, capthick=1, label='Mean ± Std')
    ax2.plot([bin_centers.min(), bin_centers.max()],
             [bin_centers.min(), bin_centers.max()],
             'r--', linewidth=2, label='y=x (perfect)')

    ax2.set_xlabel('Actual Loss (binned)')
    ax2.set_ylabel('Predicted Loss')
    ax2.set_title('Predicted Loss by Actual Loss Bins')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bdr_correlation.png', dpi=150)
    print("Saved: bdr_correlation.png")
    plt.close()


def print_statistics(data: dict) -> None:
    """Print detailed statistics."""
    predicted_loss = data['predicted_loss']
    actual_loss = data['actual_loss']
    actual_probs = data['actual_probs']
    threshold = data['threshold']

    # BDR-style: high predicted_loss > threshold = hard
    hard_mask = predicted_loss > threshold
    easy_mask = ~hard_mask

    print("=" * 60)
    print("BDR Exit Classifier Statistics")
    print("=" * 60)

    print(f"\nTotal tokens: {len(predicted_loss):,}")
    print(f"Threshold: {threshold:.4f}")

    print("\n--- Distribution Summary ---")
    print(f"  Predicted Loss: mean={predicted_loss.mean():.4f}, std={predicted_loss.std():.4f}")
    print(f"                  min={predicted_loss.min():.4f}, max={predicted_loss.max():.4f}")
    print(f"  Actual Loss:    mean={actual_loss.mean():.4f}, std={actual_loss.std():.4f}")
    print(f"                  min={actual_loss.min():.4f}, max={actual_loss.max():.4f}")

    print("\n--- Correlation ---")
    r_pearson, p_pearson = scipy_stats.pearsonr(predicted_loss, actual_loss)
    r_spearman, p_spearman = scipy_stats.spearmanr(predicted_loss, actual_loss)
    print(f"  Pearson r:  {r_pearson:.4f} (p={p_pearson:.2e})")
    print(f"  Spearman ρ: {r_spearman:.4f} (p={p_spearman:.2e})")

    print("\n--- Hard/Easy Separation ---")
    print(f"  Hard tokens: {hard_mask.sum():,} ({hard_mask.mean()*100:.1f}%)")
    print(f"  Easy tokens: {easy_mask.sum():,} ({easy_mask.mean()*100:.1f}%)")

    print("\n  Actual Loss by Category:")
    print(f"    Hard: mean={actual_loss[hard_mask].mean():.4f}, std={actual_loss[hard_mask].std():.4f}")
    print(f"    Easy: mean={actual_loss[easy_mask].mean():.4f}, std={actual_loss[easy_mask].std():.4f}")
    print(f"    Difference: {actual_loss[hard_mask].mean() - actual_loss[easy_mask].mean():.4f}")

    print("\n  Actual Probability by Category:")
    print(f"    Hard: mean={actual_probs[hard_mask].mean():.4f}")
    print(f"    Easy: mean={actual_probs[easy_mask].mean():.4f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((actual_loss[hard_mask].var() + actual_loss[easy_mask].var()) / 2)
    cohens_d = (actual_loss[hard_mask].mean() - actual_loss[easy_mask].mean()) / pooled_std
    print(f"\n  Effect Size (Cohen's d): {cohens_d:.4f}")

    if cohens_d > 0.2:
        effect_size_desc = "small" if cohens_d < 0.5 else ("medium" if cohens_d < 0.8 else "large")
        print(f"    → {effect_size_desc} effect size")

    # T-test
    t_stat, p_value = scipy_stats.ttest_ind(actual_loss[hard_mask], actual_loss[easy_mask])
    print(f"\n  T-test: t={t_stat:.2f}, p={p_value:.2e}")

    if actual_loss[hard_mask].mean() > actual_loss[easy_mask].mean() and p_value < 0.01:
        print("\n  ✓ Separation is WORKING correctly")
    else:
        print("\n  ✗ Separation may have issues")


def main() -> None:
    """Run visualization."""
    data_path = Path("analysis_data.npz")

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        print("Please run save_analysis_data.py first.")
        return

    print(f"Loading data from {data_path}...")
    data = load_data(str(data_path))
    print(f"Loaded {len(data['predicted_loss']):,} tokens\n")

    # Print statistics
    print_statistics(data)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_main_distributions(data)
    plot_correlation_analysis(data)

    print("\nDone!")


if __name__ == "__main__":
    main()
