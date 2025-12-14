"""
MLP Prediction vs Actual Loss Visualization

Visualizes the relationship between predicted loss (from MLP) and actual loss.
Generates scatter plots, histograms, and 2D density plots to find patterns.
"""

import time

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

# ============================================================
# Load data
# ============================================================
print("\nLoading data...")
data = np.load("hidden_states_data.npz")
h = torch.from_numpy(data["hidden_states"]).to(device)  # (num_tokens, dim)
loss = torch.from_numpy(data["per_token_loss"]).to(device)  # (num_tokens,)

num_tokens, dim = h.shape

print(f"Hidden states: {h.shape}")
print(f"Per-token loss: {loss.shape}")

# ============================================================
# Train MLP Router (same as analyze_mod_router.py)
# ============================================================
print(f"\n{'=' * 60}")
print("Training MLP Router (LEGO-style)")
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
    epoch_loss = 0.0
    num_batches = 0

    for i in range(0, n_train, batch_size):
        batch_h = h_train[i : i + batch_size]
        batch_target = loss_train[i : i + batch_size]

        pred = router(batch_h)
        batch_loss = F.mse_loss(pred, batch_target)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        num_batches += 1

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

    if (epoch + 1) % 10 == 0:
        print(
            f"  Epoch {epoch + 1}: train_loss={epoch_loss / num_batches:.4f}, val_loss={val_loss:.4f}"
        )

print(f"Training time: {time.time() - start_time:.1f}s")

# Get predictions on all data
router.eval()
with torch.no_grad():
    pred_loss = router(h).cpu().numpy()

actual_loss = loss.cpu().numpy()

# Statistics
corr, _ = pearsonr(pred_loss, actual_loss)
print(f"\nCorrelation (pred vs actual): {corr:.4f}")
print(f"Predicted loss: mean={pred_loss.mean():.2f}, std={pred_loss.std():.2f}")
print(f"Actual loss: mean={actual_loss.mean():.2f}, std={actual_loss.std():.2f}")

# ============================================================
# Visualization
# ============================================================
print(f"\n{'=' * 60}")
print("Creating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Scatter plot (sampled for visibility)
ax = axes[0, 0]
sample_size = min(5000, len(actual_loss))
sample_idx = np.random.choice(len(actual_loss), sample_size, replace=False)
ax.scatter(actual_loss[sample_idx], pred_loss[sample_idx], alpha=0.3, s=5)
ax.plot([0, 20], [0, 20], 'r--', linewidth=2, label='y=x (ideal)')
ax.set_xlabel('Actual Loss')
ax.set_ylabel('Predicted Loss')
ax.set_title(f'Scatter Plot (n={sample_size})\nCorr={corr:.4f}')
ax.legend()
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)

# 2. 2D Histogram (heatmap)
ax = axes[0, 1]
h2d, xedges, yedges = np.histogram2d(actual_loss, pred_loss, bins=50, range=[[0, 20], [0, 20]])
im = ax.imshow(h2d.T, origin='lower', aspect='auto',
               extent=[0, 20, 0, 20], cmap='hot')
ax.plot([0, 20], [0, 20], 'cyan', linewidth=2, linestyle='--', label='y=x')
ax.set_xlabel('Actual Loss')
ax.set_ylabel('Predicted Loss')
ax.set_title('2D Density (Heat Map)')
ax.legend()
plt.colorbar(im, ax=ax, label='Count')

# 3. Histograms
ax = axes[0, 2]
ax.hist(actual_loss, bins=100, alpha=0.7, label='Actual Loss', density=True)
ax.hist(pred_loss, bins=100, alpha=0.7, label='Predicted Loss', density=True)
ax.set_xlabel('Loss')
ax.set_ylabel('Density')
ax.set_title('Distribution Comparison')
ax.legend()
ax.set_xlim(0, 20)

# 4. Residual plot (pred - actual)
ax = axes[1, 0]
residual = pred_loss - actual_loss
ax.scatter(actual_loss[sample_idx], residual[sample_idx], alpha=0.3, s=5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Actual Loss')
ax.set_ylabel('Residual (Pred - Actual)')
ax.set_title(f'Residual Plot\nMean={residual.mean():.2f}, Std={residual.std():.2f}')
ax.set_xlim(0, 20)

# 5. Binned mean prediction
ax = axes[1, 1]
n_bins = 20
bin_edges = np.linspace(0, 20, n_bins + 1)
bin_means_actual = []
bin_means_pred = []
bin_stds_pred = []
bin_centers = []

for i in range(n_bins):
    mask = (actual_loss >= bin_edges[i]) & (actual_loss < bin_edges[i + 1])
    if mask.sum() > 10:
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_means_actual.append(actual_loss[mask].mean())
        bin_means_pred.append(pred_loss[mask].mean())
        bin_stds_pred.append(pred_loss[mask].std())

bin_centers = np.array(bin_centers)
bin_means_actual = np.array(bin_means_actual)
bin_means_pred = np.array(bin_means_pred)
bin_stds_pred = np.array(bin_stds_pred)

ax.errorbar(bin_centers, bin_means_pred, yerr=bin_stds_pred, fmt='o-', capsize=3, label='Pred (mean ± std)')
ax.plot(bin_centers, bin_means_actual, 'r--', linewidth=2, label='Actual (y=x)')
ax.set_xlabel('Actual Loss (binned)')
ax.set_ylabel('Predicted Loss')
ax.set_title('Binned Mean Prediction')
ax.legend()
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)

# 6. Easy/Hard separation quality
ax = axes[1, 2]
threshold = np.median(pred_loss)
easy_mask = pred_loss <= threshold
hard_mask = ~easy_mask

# Box plot for easy vs hard
box_data = [actual_loss[easy_mask], actual_loss[hard_mask]]
bp = ax.boxplot(box_data, labels=['Easy\n(pred_loss <= median)', 'Hard\n(pred_loss > median)'])
ax.set_ylabel('Actual Loss')
ax.set_title('Easy/Hard Separation Quality')

# Add statistics
easy_mean = actual_loss[easy_mask].mean()
hard_mean = actual_loss[hard_mask].mean()
ax.axhline(y=easy_mean, color='blue', linestyle='--', alpha=0.5)
ax.axhline(y=hard_mean, color='orange', linestyle='--', alpha=0.5)
ax.text(0.95, 0.95, f'Easy mean: {easy_mean:.2f}\nHard mean: {hard_mean:.2f}\nDiff: {hard_mean - easy_mean:.2f}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('mlp_prediction_analysis.png', dpi=150)
print("Saved: mlp_prediction_analysis.png")

# ============================================================
# Additional Analysis: By loss range
# ============================================================
print(f"\n{'=' * 60}")
print("Analysis by Loss Range")
print("=" * 60)

print(f"\n{'Loss Range':<15} {'N tokens':<12} {'Mean Actual':<12} {'Mean Pred':<12} {'Mean Error':<12} {'Corr':<10}")
print("-" * 73)

ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 15), (15, 20)]
for low, high in ranges:
    mask = (actual_loss >= low) & (actual_loss < high)
    n = mask.sum()
    if n > 100:
        mean_actual = actual_loss[mask].mean()
        mean_pred = pred_loss[mask].mean()
        mean_error = (pred_loss[mask] - actual_loss[mask]).mean()
        corr_range, _ = pearsonr(pred_loss[mask], actual_loss[mask])
        print(f"{low}-{high:<11} {n:<12} {mean_actual:<12.2f} {mean_pred:<12.2f} {mean_error:<12.2f} {corr_range:<10.4f}")
    else:
        print(f"{low}-{high:<11} {n:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

# ============================================================
# Key Findings
# ============================================================
print(f"\n{'=' * 60}")
print("Key Findings")
print("=" * 60)

# Calculate oracle percentage
oracle_threshold = np.median(actual_loss)
oracle_easy = actual_loss <= oracle_threshold
oracle_diff = actual_loss[~oracle_easy].mean() - actual_loss[oracle_easy].mean()
mlp_diff = hard_mean - easy_mean
oracle_pct = (mlp_diff / oracle_diff) * 100

print(f"""
1. Overall Correlation: {corr:.4f}
   - MLP can predict loss with moderate correlation
   - Not perfect, but captures the trend

2. Easy/Hard Separation:
   - Easy tokens (pred <= median): actual_loss = {easy_mean:.2f}
   - Hard tokens (pred > median): actual_loss = {hard_mean:.2f}
   - Diff: {mlp_diff:.2f}
   - Oracle %: {oracle_pct:.1f}%

3. Prediction Bias:
   - Mean residual: {residual.mean():.2f}
   - Std residual: {residual.std():.2f}
   - {"MLP tends to underestimate high losses" if residual.mean() < 0 else "MLP tends to overestimate"}

4. Visual Patterns:
   - Check mlp_prediction_analysis.png for detailed plots
""")

plt.show()
