"""
Analyze Saved Data

Load analysis_data.npz and perform various analyses.
No training required - just load and analyze.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from pathlib import Path


def load_data(path: str = "analysis_data.npz") -> dict:
    """Load saved analysis data."""
    data = np.load(path)
    return {
        'confidences': data['confidences'],
        'actual_probs': data['actual_probs'],
        'per_token_loss': data['per_token_loss'],
        'exit_labels': data['exit_labels'],
        'threshold': float(data['threshold']),
        'best_val_ppl': float(data['best_val_ppl']),
    }


def analyze_hard_easy_separation(data: dict) -> None:
    """Analyze if Hard/Easy separation is working."""
    confidences = data['confidences']
    per_token_loss = data['per_token_loss']
    actual_probs = data['actual_probs']
    exit_labels = data['exit_labels']
    threshold = data['threshold']

    hard_mask = confidences < threshold
    easy_mask = ~hard_mask

    print("=" * 60)
    print("Hard/Easy Separation Analysis")
    print("=" * 60)

    print(f"\nThreshold: {threshold:.4f}")
    print(f"Hard tokens: {hard_mask.sum():,} ({hard_mask.mean()*100:.1f}%)")
    print(f"Easy tokens: {easy_mask.sum():,} ({easy_mask.mean()*100:.1f}%)")

    print("\n--- Per-Token Loss ---")
    print(f"  Hard: mean={per_token_loss[hard_mask].mean():.4f}, std={per_token_loss[hard_mask].std():.4f}")
    print(f"  Easy: mean={per_token_loss[easy_mask].mean():.4f}, std={per_token_loss[easy_mask].std():.4f}")
    print(f"  Difference: {per_token_loss[hard_mask].mean() - per_token_loss[easy_mask].mean():.4f}")

    print("\n--- Actual Probability ---")
    print(f"  Hard: mean={actual_probs[hard_mask].mean():.4f}, std={actual_probs[hard_mask].std():.4f}")
    print(f"  Easy: mean={actual_probs[easy_mask].mean():.4f}, std={actual_probs[easy_mask].std():.4f}")

    print("\n--- Exit Labels (exp(-loss)) ---")
    print(f"  Hard: mean={exit_labels[hard_mask].mean():.4f}, std={exit_labels[hard_mask].std():.4f}")
    print(f"  Easy: mean={exit_labels[easy_mask].mean():.4f}, std={exit_labels[easy_mask].std():.4f}")

    # Statistical test
    t_stat, p_value = scipy_stats.ttest_ind(per_token_loss[hard_mask], per_token_loss[easy_mask])
    print(f"\n--- T-test (Hard vs Easy Loss) ---")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.2e}")

    if per_token_loss[hard_mask].mean() > per_token_loss[easy_mask].mean() and p_value < 0.01:
        print("\n  ✓ Separation is statistically significant and in expected direction")
    else:
        print("\n  ✗ Separation may not be working as expected")


def plot_hard_easy_comparison(data: dict) -> None:
    """Create visualization comparing Hard and Easy tokens."""
    confidences = data['confidences']
    per_token_loss = data['per_token_loss']
    actual_probs = data['actual_probs']
    threshold = data['threshold']

    hard_mask = confidences < threshold
    easy_mask = ~hard_mask

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Loss distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(per_token_loss[hard_mask], bins=50, alpha=0.6, label=f'Hard (n={hard_mask.sum():,})', color='red', density=True)
    ax1.hist(per_token_loss[easy_mask], bins=50, alpha=0.6, label=f'Easy (n={easy_mask.sum():,})', color='green', density=True)
    ax1.axvline(per_token_loss[hard_mask].mean(), color='darkred', linestyle='--', label=f'Hard mean={per_token_loss[hard_mask].mean():.2f}')
    ax1.axvline(per_token_loss[easy_mask].mean(), color='darkgreen', linestyle='--', label=f'Easy mean={per_token_loss[easy_mask].mean():.2f}')
    ax1.set_xlabel('Per-Token Loss')
    ax1.set_ylabel('Density')
    ax1.set_title('Loss Distribution: Hard vs Easy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Actual probability distribution comparison
    ax2 = axes[0, 1]
    ax2.hist(actual_probs[hard_mask], bins=50, alpha=0.6, label='Hard', color='red', density=True)
    ax2.hist(actual_probs[easy_mask], bins=50, alpha=0.6, label='Easy', color='green', density=True)
    ax2.set_xlabel('Actual Probability (softmax[correct])')
    ax2.set_ylabel('Density')
    ax2.set_title('Actual Probability Distribution: Hard vs Easy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Confidence vs Loss scatter
    ax3 = axes[1, 0]
    # Subsample for scatter plot (too many points)
    n_sample = min(5000, len(confidences))
    idx = np.random.choice(len(confidences), n_sample, replace=False)
    scatter = ax3.scatter(confidences[idx], per_token_loss[idx], c=per_token_loss[idx],
                          cmap='RdYlGn_r', alpha=0.3, s=5)
    ax3.axvline(threshold, color='purple', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Per-Token Loss')
    ax3.set_title('Confidence vs Loss (sampled)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Loss')

    # 4. Boxplot comparison
    ax4 = axes[1, 1]
    box_data = [per_token_loss[hard_mask], per_token_loss[easy_mask]]
    bp = ax4.boxplot(box_data, labels=['Hard', 'Easy'], patch_artist=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('green')
    bp['boxes'][1].set_alpha(0.5)
    ax4.set_ylabel('Per-Token Loss')
    ax4.set_title('Loss Distribution Boxplot')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hard_easy_separation.png', dpi=150)
    print("\nSaved: hard_easy_separation.png")
    plt.show()


def analyze_confidence_calibration(data: dict) -> None:
    """Analyze confidence vs actual probability correlation."""
    confidences = data['confidences']
    actual_probs = data['actual_probs']
    exit_labels = data['exit_labels']

    print("\n" + "=" * 60)
    print("Confidence Calibration Analysis")
    print("=" * 60)

    # Correlation with actual_probs
    pearson_r, pearson_p = scipy_stats.pearsonr(confidences, actual_probs)
    spearman_r, spearman_p = scipy_stats.spearmanr(confidences, actual_probs)

    print("\n--- Confidence vs Actual Probability ---")
    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")

    # Correlation with exit_labels (what it was trained on)
    pearson_r2, pearson_p2 = scipy_stats.pearsonr(confidences, exit_labels)
    spearman_r2, spearman_p2 = scipy_stats.spearmanr(confidences, exit_labels)

    print("\n--- Confidence vs Exit Labels (training target) ---")
    print(f"  Pearson r: {pearson_r2:.4f} (p={pearson_p2:.2e})")
    print(f"  Spearman ρ: {spearman_r2:.4f} (p={spearman_p2:.2e})")

    print("\n--- Scale Comparison ---")
    print(f"  Confidence:   mean={confidences.mean():.4f}, std={confidences.std():.4f}")
    print(f"  Exit Labels:  mean={exit_labels.mean():.4f}, std={exit_labels.std():.4f}")
    print(f"  Actual Probs: mean={actual_probs.mean():.4f}, std={actual_probs.std():.4f}")


def main() -> None:
    """Run all analyses."""
    data_path = Path("analysis_data.npz")

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        print("Please run save_analysis_data.py first.")
        return

    print(f"Loading data from {data_path}...")
    data = load_data(str(data_path))
    print(f"Loaded {len(data['confidences']):,} tokens")

    # Run analyses
    analyze_hard_easy_separation(data)
    analyze_confidence_calibration(data)
    plot_hard_easy_comparison(data)


if __name__ == "__main__":
    main()
