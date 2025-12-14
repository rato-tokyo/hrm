"""
Confidence vs Actual Prediction Probability Correlation Analysis

Compares exit_classifier confidence with actual softmax probability of correct token.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

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
    """Compare confidence with actual prediction probability."""
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
    print("Confidence vs Actual Probability Correlation Analysis")
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

    # Compute both confidence and actual probability on validation data
    print("\nComputing confidence and actual probabilities on validation data...")
    block.eval()

    all_confidences: list[torch.Tensor] = []
    all_actual_probs: list[torch.Tensor] = []

    with torch.no_grad():
        for h, y in val_data.to(str(device)).batches(trainer_config.batch_size, shuffle=False):
            h_out, logits, _ = block.forward(h)

            # Exit classifier confidence
            confidence = block.exit_classifier.compute_confidence(h_out)  # (batch, seq_len)

            # Actual probability of correct token = exp(-cross_entropy_loss)
            probs = torch.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
            # Gather probability of correct token
            actual_prob = probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)

            all_confidences.append(confidence.cpu())
            all_actual_probs.append(actual_prob.cpu())

    confidences_flat = torch.cat(all_confidences).view(-1).numpy()
    actual_probs_flat = torch.cat(all_actual_probs).view(-1).numpy()

    print(f"Total tokens: {len(confidences_flat)}")

    # Compute correlation metrics
    pearson_r, pearson_p = scipy_stats.pearsonr(confidences_flat, actual_probs_flat)
    spearman_r, spearman_p = scipy_stats.spearmanr(confidences_flat, actual_probs_flat)

    # Compute R² (coefficient of determination)
    ss_res = np.sum((actual_probs_flat - confidences_flat) ** 2)
    ss_tot = np.sum((actual_probs_flat - actual_probs_flat.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Mean Absolute Error
    mae = np.mean(np.abs(confidences_flat - actual_probs_flat))

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((confidences_flat - actual_probs_flat) ** 2))

    print("\n" + "=" * 60)
    print("Correlation Metrics")
    print("=" * 60)
    print(f"  Pearson correlation:  r = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"  Spearman correlation: ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")
    print(f"  R² (if perfect match): {r_squared:.4f}")
    print(f"  Mean Absolute Error:   {mae:.4f}")
    print(f"  Root Mean Square Error: {rmse:.4f}")

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"  Confidence   - Mean: {confidences_flat.mean():.4f}, Std: {confidences_flat.std():.4f}")
    print(f"  Actual Prob  - Mean: {actual_probs_flat.mean():.4f}, Std: {actual_probs_flat.std():.4f}")

    # Create scatter plot with density
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Scatter plot with density coloring
    ax1 = axes[0]

    # Use hexbin for density visualization (better for many points)
    hb = ax1.hexbin(
        actual_probs_flat, confidences_flat,
        gridsize=50, cmap='YlOrRd', mincnt=1,
    )
    plt.colorbar(hb, ax=ax1, label='Count')

    # Add diagonal line (perfect correlation)
    ax1.plot([0, 1], [0, 1], 'b--', linewidth=2, label='Perfect match (y=x)')

    # Add regression line
    slope, intercept, _, _, _ = scipy_stats.linregress(actual_probs_flat, confidences_flat)
    x_line = np.linspace(0, 1, 100)
    ax1.plot(x_line, slope * x_line + intercept, 'g-', linewidth=2,
             label=f'Regression: y={slope:.2f}x+{intercept:.2f}')

    ax1.set_xlabel('Actual Probability (softmax[correct])', fontsize=11)
    ax1.set_ylabel('Exit Classifier Confidence', fontsize=11)
    ax1.set_title(f'Confidence vs Actual Probability\nr={pearson_r:.3f}, ρ={spearman_r:.3f}', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of differences
    ax2 = axes[1]
    differences = confidences_flat - actual_probs_flat

    ax2.hist(differences, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax2.axvline(x=differences.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean = {differences.mean():.4f}')

    ax2.set_xlabel('Confidence - Actual Probability', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'Distribution of Differences\nMAE={mae:.4f}, RMSE={rmse:.4f}', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Binned comparison
    ax3 = axes[2]

    # Bin actual probabilities and compute mean confidence for each bin
    num_bins = 20
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(actual_probs_flat, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_means = []
    bin_stds = []
    bin_counts = []
    for i in range(num_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(confidences_flat[mask].mean())
            bin_stds.append(confidences_flat[mask].std())
            bin_counts.append(mask.sum())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    # Plot binned means with error bars
    valid = ~np.isnan(bin_means)
    ax3.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_stds[valid],
                 fmt='o-', capsize=3, color='blue', label='Mean ± Std')
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')

    ax3.set_xlabel('Actual Probability (binned)', fontsize=11)
    ax3.set_ylabel('Mean Confidence', fontsize=11)
    ax3.set_title('Calibration Plot\n(Binned Analysis)', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('confidence_correlation.png', dpi=150)
    print("\nSaved: confidence_correlation.png")
    plt.show()

    # Interpretation
    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)

    if pearson_r > 0.9:
        quality = "Excellent"
    elif pearson_r > 0.7:
        quality = "Good"
    elif pearson_r > 0.5:
        quality = "Moderate"
    else:
        quality = "Weak"

    print(f"  Correlation quality: {quality}")
    print()

    if differences.mean() > 0.05:
        print("  → Exit classifier is OVERCONFIDENT (predicts higher than actual)")
    elif differences.mean() < -0.05:
        print("  → Exit classifier is UNDERCONFIDENT (predicts lower than actual)")
    else:
        print("  → Exit classifier is well-calibrated (close to actual)")

    print()
    print("  Note: Exit classifier learns to predict exp(-loss) from hidden states alone,")
    print("        without access to the target token. High correlation means it can")
    print("        successfully estimate prediction difficulty from the hidden representation.")


if __name__ == "__main__":
    main()
