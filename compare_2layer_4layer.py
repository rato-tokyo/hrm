"""
Compare 2-layer vs 4-layer MLP Prediction Performance

Compares:
1. Loss distribution differences
2. MLP prediction accuracy (correlation)
3. Easy/Hard separation quality (Oracle %)

Requires:
- hidden_states_data.npz (2-layer)
- hidden_states_data_4layer.npz (4-layer)
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr


print("=" * 60)
if torch.cuda.is_available():
    device = "cuda"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = "mps"
    print(f"Device: {device}")
else:
    device = "cpu"
    print(f"Device: {device}")
print("=" * 60)


class MLPRouter(nn.Module):
    """2-layer MLP router."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(h))
        return self.fc2(x).squeeze(-1)


def train_and_evaluate(h: torch.Tensor, loss: torch.Tensor, name: str) -> dict:
    """Train MLP and evaluate prediction quality."""
    print(f"\n{'=' * 60}")
    print(f"Training MLP Router for {name}")
    print("=" * 60)

    num_tokens, dim = h.shape

    # Train/test split
    torch.manual_seed(42)
    n_train = int(num_tokens * 0.8)
    perm = torch.randperm(num_tokens)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    h_train = h[train_idx]
    h_test = h[test_idx]
    loss_train = loss[train_idx]
    loss_test = loss[test_idx]

    print(f"Train: {n_train}, Test: {num_tokens - n_train}")

    # Train MLP
    router = MLPRouter(dim, 128).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)

    batch_size = 4096
    num_epochs = 50
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()
    for epoch in range(num_epochs):
        router.train()
        for i in range(0, n_train, batch_size):
            batch_h = h_train[i : i + batch_size]
            batch_target = loss_train[i : i + batch_size]

            pred = router(batch_h)
            batch_loss = F.mse_loss(pred, batch_target)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # Validation
        router.eval()
        with torch.no_grad():
            val_pred = router(h_test)
            val_loss = F.mse_loss(val_pred, loss_test).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"Training time: {time.time() - start_time:.1f}s")

    # Get predictions on all data
    router.eval()
    with torch.no_grad():
        pred_loss = router(h).cpu().numpy()

    actual_loss = loss.cpu().numpy()

    # Compute metrics
    corr, _ = pearsonr(pred_loss, actual_loss)

    # Easy/Hard separation
    threshold = np.median(pred_loss)
    easy_mask = pred_loss <= threshold
    hard_mask = ~easy_mask

    easy_actual = actual_loss[easy_mask].mean()
    hard_actual = actual_loss[hard_mask].mean()
    diff = hard_actual - easy_actual

    # Oracle
    oracle_threshold = np.median(actual_loss)
    oracle_easy = actual_loss <= oracle_threshold
    oracle_diff = actual_loss[~oracle_easy].mean() - actual_loss[oracle_easy].mean()
    oracle_pct = (diff / oracle_diff) * 100

    print(f"\nResults:")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Easy tokens actual loss: {easy_actual:.2f}")
    print(f"  Hard tokens actual loss: {hard_actual:.2f}")
    print(f"  Diff: {diff:.2f}")
    print(f"  Oracle %: {oracle_pct:.1f}%")

    return {
        "name": name,
        "pred_loss": pred_loss,
        "actual_loss": actual_loss,
        "corr": corr,
        "easy_loss": easy_actual,
        "hard_loss": hard_actual,
        "diff": diff,
        "oracle_pct": oracle_pct,
        "oracle_diff": oracle_diff,
    }


def main():
    # Check files exist
    path_2layer = Path("hidden_states_data.npz")
    path_4layer = Path("hidden_states_data_4layer.npz")

    if not path_2layer.exists():
        print(f"ERROR: {path_2layer} not found. Run save_hidden_states_simple.py first.")
        return

    if not path_4layer.exists():
        print(f"ERROR: {path_4layer} not found. Run save_hidden_states_4layer.py first.")
        return

    # Load data
    print("\nLoading 2-layer data...")
    data_2layer = np.load(path_2layer)
    h_2layer = torch.from_numpy(data_2layer["hidden_states"]).to(device)
    loss_2layer = torch.from_numpy(data_2layer["per_token_loss"]).to(device)
    print(f"  Shape: {h_2layer.shape}, Loss mean: {loss_2layer.mean():.2f}")

    print("\nLoading 4-layer data...")
    data_4layer = np.load(path_4layer)
    h_4layer = torch.from_numpy(data_4layer["hidden_states"]).to(device)
    loss_4layer = torch.from_numpy(data_4layer["per_token_loss"]).to(device)
    print(f"  Shape: {h_4layer.shape}, Loss mean: {loss_4layer.mean():.2f}")

    # Train and evaluate both
    result_2layer = train_and_evaluate(h_2layer, loss_2layer, "2-layer")
    result_4layer = train_and_evaluate(h_4layer, loss_4layer, "4-layer")

    # ============================================================
    # Summary Comparison
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Summary Comparison: 2-layer vs 4-layer")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'2-layer':<15} {'4-layer':<15} {'Diff':<15}")
    print("-" * 70)
    print(f"{'Actual Loss (mean)':<25} {result_2layer['actual_loss'].mean():<15.2f} {result_4layer['actual_loss'].mean():<15.2f} {result_4layer['actual_loss'].mean() - result_2layer['actual_loss'].mean():<+15.2f}")
    print(f"{'Actual Loss (std)':<25} {result_2layer['actual_loss'].std():<15.2f} {result_4layer['actual_loss'].std():<15.2f} {result_4layer['actual_loss'].std() - result_2layer['actual_loss'].std():<+15.2f}")
    print(f"{'MLP Correlation':<25} {result_2layer['corr']:<15.4f} {result_4layer['corr']:<15.4f} {result_4layer['corr'] - result_2layer['corr']:<+15.4f}")
    print(f"{'Easy/Hard Diff':<25} {result_2layer['diff']:<15.2f} {result_4layer['diff']:<15.2f} {result_4layer['diff'] - result_2layer['diff']:<+15.2f}")
    print(f"{'Oracle %':<25} {result_2layer['oracle_pct']:<15.1f}% {result_4layer['oracle_pct']:<15.1f}% {result_4layer['oracle_pct'] - result_2layer['oracle_pct']:<+15.1f}%")
    print(f"{'Oracle Diff (reference)':<25} {result_2layer['oracle_diff']:<15.2f} {result_4layer['oracle_diff']:<15.2f}")

    # ============================================================
    # Visualization
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Creating Comparison Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: 2-layer plots
    # Scatter
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(result_2layer['actual_loss']), 3000, replace=False)
    ax.scatter(result_2layer['actual_loss'][sample_idx], result_2layer['pred_loss'][sample_idx], alpha=0.3, s=5)
    ax.plot([0, 20], [0, 20], 'r--', linewidth=2)
    ax.set_xlabel('Actual Loss')
    ax.set_ylabel('Predicted Loss')
    ax.set_title(f"2-layer: Corr={result_2layer['corr']:.3f}, Oracle={result_2layer['oracle_pct']:.1f}%")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    # Distribution
    ax = axes[0, 1]
    ax.hist(result_2layer['actual_loss'], bins=50, alpha=0.7, label='Actual', density=True)
    ax.hist(result_2layer['pred_loss'], bins=50, alpha=0.7, label='Predicted', density=True)
    ax.set_xlabel('Loss')
    ax.set_ylabel('Density')
    ax.set_title(f"2-layer: Loss Distribution")
    ax.legend()
    ax.set_xlim(0, 20)

    # Binned mean
    ax = axes[0, 2]
    bin_edges = np.linspace(0, 20, 21)
    bin_centers = []
    bin_means = []
    for i in range(len(bin_edges) - 1):
        mask = (result_2layer['actual_loss'] >= bin_edges[i]) & (result_2layer['actual_loss'] < bin_edges[i + 1])
        if mask.sum() > 10:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(result_2layer['pred_loss'][mask].mean())
    ax.plot(bin_centers, bin_means, 'bo-', label='Predicted')
    ax.plot([0, 20], [0, 20], 'r--', label='Ideal')
    ax.set_xlabel('Actual Loss (binned)')
    ax.set_ylabel('Mean Predicted Loss')
    ax.set_title('2-layer: Binned Mean Prediction')
    ax.legend()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    # Row 2: 4-layer plots
    # Scatter
    ax = axes[1, 0]
    sample_idx = np.random.choice(len(result_4layer['actual_loss']), 3000, replace=False)
    ax.scatter(result_4layer['actual_loss'][sample_idx], result_4layer['pred_loss'][sample_idx], alpha=0.3, s=5)
    ax.plot([0, 20], [0, 20], 'r--', linewidth=2)
    ax.set_xlabel('Actual Loss')
    ax.set_ylabel('Predicted Loss')
    ax.set_title(f"4-layer: Corr={result_4layer['corr']:.3f}, Oracle={result_4layer['oracle_pct']:.1f}%")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    # Distribution
    ax = axes[1, 1]
    ax.hist(result_4layer['actual_loss'], bins=50, alpha=0.7, label='Actual', density=True)
    ax.hist(result_4layer['pred_loss'], bins=50, alpha=0.7, label='Predicted', density=True)
    ax.set_xlabel('Loss')
    ax.set_ylabel('Density')
    ax.set_title(f"4-layer: Loss Distribution")
    ax.legend()
    ax.set_xlim(0, 20)

    # Binned mean
    ax = axes[1, 2]
    bin_centers = []
    bin_means = []
    for i in range(len(bin_edges) - 1):
        mask = (result_4layer['actual_loss'] >= bin_edges[i]) & (result_4layer['actual_loss'] < bin_edges[i + 1])
        if mask.sum() > 10:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(result_4layer['pred_loss'][mask].mean())
    ax.plot(bin_centers, bin_means, 'bo-', label='Predicted')
    ax.plot([0, 20], [0, 20], 'r--', label='Ideal')
    ax.set_xlabel('Actual Loss (binned)')
    ax.set_ylabel('Mean Predicted Loss')
    ax.set_title('4-layer: Binned Mean Prediction')
    ax.legend()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig('compare_2layer_4layer.png', dpi=150)
    print("Saved: compare_2layer_4layer.png")

    # ============================================================
    # Key Insights
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Key Insights")
    print("=" * 60)

    loss_diff = result_4layer['actual_loss'].mean() - result_2layer['actual_loss'].mean()
    oracle_diff = result_4layer['oracle_pct'] - result_2layer['oracle_pct']

    print(f"""
1. Loss Comparison:
   - 2-layer mean loss: {result_2layer['actual_loss'].mean():.2f}
   - 4-layer mean loss: {result_4layer['actual_loss'].mean():.2f}
   - {'4-layer has LOWER loss (better model)' if loss_diff < 0 else '4-layer has HIGHER loss'}

2. MLP Prediction Quality:
   - 2-layer correlation: {result_2layer['corr']:.4f}
   - 4-layer correlation: {result_4layer['corr']:.4f}
   - {'4-layer predictions are MORE accurate' if result_4layer['corr'] > result_2layer['corr'] else '2-layer predictions are MORE accurate'}

3. Easy/Hard Separation:
   - 2-layer Oracle %: {result_2layer['oracle_pct']:.1f}%
   - 4-layer Oracle %: {result_4layer['oracle_pct']:.1f}%
   - {'4-layer separation is BETTER' if oracle_diff > 0 else '2-layer separation is BETTER'}

4. Hypothesis:
   - If 4-layer has lower loss: deeper model learns better representations
   - If 4-layer has higher Oracle %: hidden states encode more predictive info
   - If correlation improves: deeper model provides richer features for MLP
""")

    plt.show()


if __name__ == "__main__":
    main()
