"""
Low-Rank Approximation and Alternative Functions Analysis (GPU Version)

Experiments:
1. Low-rank approximation accuracy for softmax confidence
2. Comparison of different functions for predictability from hidden states
   - softmax_confidence
   - max(z)
   - max(z) - mean(z)
   - margin (max - second_max)

GPU-accelerated for fast processing on Colab.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import time

# ============================================================
# Setup Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ============================================================
# Load Data
# ============================================================
print("\nLoading data...")

data = np.load("hidden_states_data.npz")
h_np = data["hidden_states"]  # (num_tokens, dim)
loss_np = data["per_token_loss"]  # (num_tokens,)
W_np = data["output_head_W"]  # (vocab_size, dim)

num_tokens, dim = h_np.shape
vocab_size = W_np.shape[0]

print(f"Hidden states: {h_np.shape}")
print(f"Per-token loss: {loss_np.shape}")
print(f"W matrix: {W_np.shape}")

# Move to GPU
h = torch.from_numpy(h_np).float().to(device)
W = torch.from_numpy(W_np).float().to(device)
loss = torch.from_numpy(loss_np).float().to(device)

print(f"Data moved to {device}")

# ============================================================
# Compute Confidence Metrics (GPU)
# ============================================================
print(f"\n{'=' * 60}")
print("Computing confidence metrics (GPU)...")
print("=" * 60)

chunk_size = 4000  # Larger chunks since GPU has more memory
num_chunks = (num_tokens + chunk_size - 1) // chunk_size

softmax_conf = torch.zeros(num_tokens, device=device)
max_z = torch.zeros(num_tokens, device=device)
max_minus_mean = torch.zeros(num_tokens, device=device)
margin = torch.zeros(num_tokens, device=device)

start_time = time.time()
print(f"Processing {num_chunks} chunks (chunk_size={chunk_size})...")

with torch.no_grad():
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_tokens)

        # z_chunk: (chunk_size, vocab_size)
        z_chunk = h[start:end] @ W.T

        # Softmax confidence (max probability)
        probs = F.softmax(z_chunk, dim=1)
        softmax_conf[start:end] = probs.max(dim=1).values

        # Max logit
        max_z[start:end] = z_chunk.max(dim=1).values

        # Max - mean
        max_minus_mean[start:end] = z_chunk.max(dim=1).values - z_chunk.mean(dim=1)

        # Margin (max - second_max)
        top2 = z_chunk.topk(2, dim=1).values
        margin[start:end] = top2[:, 0] - top2[:, 1]

        # Progress
        if (i + 1) % 5 == 0 or i == num_chunks - 1:
            elapsed = time.time() - start_time
            progress = (i + 1) / num_chunks
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  [{i + 1:3d}/{num_chunks}] {progress * 100:.1f}% done, elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

print(f"Done! Total time: {time.time() - start_time:.1f}s")

# ============================================================
# Experiment 1: Low-Rank Approximation
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 1: Low-Rank Approximation")
print("=" * 60)

# SVD of W (on CPU, then move results to GPU)
print("Computing SVD of W...")
svd_start = time.time()
U_np, S_np, Vt_np = np.linalg.svd(W_np, full_matrices=False)
print(f"SVD done in {time.time() - svd_start:.1f}s")
print(f"SVD shapes: U={U_np.shape}, S={S_np.shape}, Vt={Vt_np.shape}")

# Move to GPU
U = torch.from_numpy(U_np).float().to(device)
S = torch.from_numpy(S_np).float().to(device)
Vt = torch.from_numpy(Vt_np).float().to(device)

# Sanity check: verify SVD reconstruction
print("\nSanity check: SVD reconstruction...")
W_reconstructed = U @ torch.diag(S) @ Vt
reconstruction_error = torch.norm(W - W_reconstructed).item() / torch.norm(W).item()
print(f"  Relative reconstruction error: {reconstruction_error:.2e}")

# Check logits reconstruction for a sample
with torch.no_grad():
    h_sample = h[:100]
    z_exact = h_sample @ W.T
    z_reconstructed = h_sample @ W_reconstructed.T
    logits_corr = torch.corrcoef(torch.stack([z_exact.flatten(), z_reconstructed.flatten()]))[0, 1].item()
    print(f"  Logits correlation (full rank): {logits_corr:.6f}")

# Test different ranks
ranks = [1, 2, 5, 10, 20, 32, 64]
print(f"\n{'Rank':<6} {'Coverage':<10} {'Corr':<12} {'Logits Corr':<12} {'MSE':<15} {'Time':<10}")
print("-" * 75)

for r in ranks:
    rank_start = time.time()

    # Precompute low-rank components
    Vt_r = Vt[:r, :]  # (r, dim)
    S_r = S[:r]  # (r,)
    U_r = U[:, :r]  # (vocab_size, r)

    softmax_conf_approx = torch.zeros(num_tokens, device=device)

    # Also track logits correlation for debugging
    logits_corr_sum = 0.0
    logits_count = 0

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_tokens)

            h_chunk = h[start:end]  # (chunk, dim)

            # z_approx = h @ Vt_r.T @ diag(S_r) @ U_r.T
            h_proj = h_chunk @ Vt_r.T  # (chunk, r)
            h_scaled = h_proj * S_r  # (chunk, r)
            z_approx = h_scaled @ U_r.T  # (chunk, vocab_size)

            # Softmax confidence
            probs = F.softmax(z_approx, dim=1)
            softmax_conf_approx[start:end] = probs.max(dim=1).values

            # Track logits correlation (for first few chunks only)
            if i < 3:
                z_exact_chunk = h_chunk @ W.T
                chunk_corr = torch.corrcoef(torch.stack([z_exact_chunk.flatten(), z_approx.flatten()]))[0, 1].item()
                logits_corr_sum += chunk_corr
                logits_count += 1

    # Coverage
    coverage = (S[:r] ** 2).sum().item() / (S ** 2).sum().item() * 100

    # Average logits correlation
    avg_logits_corr = logits_corr_sum / logits_count if logits_count > 0 else 0.0

    # Move to CPU for correlation
    conf_exact_cpu = softmax_conf.cpu().numpy()
    conf_approx_cpu = softmax_conf_approx.cpu().numpy()

    corr, _ = pearsonr(conf_approx_cpu, conf_exact_cpu)
    mse = np.mean((conf_approx_cpu - conf_exact_cpu) ** 2)

    elapsed = time.time() - rank_start
    print(f"{r:<6} {coverage:<10.1f}% {corr:<12.4f} {avg_logits_corr:<12.4f} {mse:<15.6f} {elapsed:<10.1f}s")

# ============================================================
# Experiment 1b: Low-Rank for Logit-based Metrics (no softmax)
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 1b: Low-Rank for Logit-based Metrics")
print("=" * 60)
print("(softmaxを使わず、logitsの統計量を直接近似)")

print(f"\n{'Rank':<6} {'Coverage':<10} {'max_z Corr':<12} {'max-mean Corr':<14} {'margin Corr':<12}")
print("-" * 70)

for r in ranks:
    rank_start = time.time()

    # Precompute low-rank components
    Vt_r = Vt[:r, :]  # (r, dim)
    S_r = S[:r]  # (r,)
    U_r = U[:, :r]  # (vocab_size, r)

    max_z_approx = torch.zeros(num_tokens, device=device)
    max_minus_mean_approx = torch.zeros(num_tokens, device=device)
    margin_approx = torch.zeros(num_tokens, device=device)

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_tokens)

            h_chunk = h[start:end]  # (chunk, dim)

            # z_approx = h @ Vt_r.T @ diag(S_r) @ U_r.T
            h_proj = h_chunk @ Vt_r.T  # (chunk, r)
            h_scaled = h_proj * S_r  # (chunk, r)
            z_approx = h_scaled @ U_r.T  # (chunk, vocab_size)

            # Max logit
            max_z_approx[start:end] = z_approx.max(dim=1).values

            # Max - mean
            max_minus_mean_approx[start:end] = z_approx.max(dim=1).values - z_approx.mean(dim=1)

            # Margin (max - second_max)
            top2 = z_approx.topk(2, dim=1).values
            margin_approx[start:end] = top2[:, 0] - top2[:, 1]

    # Coverage
    coverage = (S[:r] ** 2).sum().item() / (S ** 2).sum().item() * 100

    # Correlations with exact values
    max_z_corr, _ = pearsonr(max_z_approx.cpu().numpy(), max_z_cpu)
    max_minus_mean_corr, _ = pearsonr(max_minus_mean_approx.cpu().numpy(), max_minus_mean_cpu)
    margin_corr, _ = pearsonr(margin_approx.cpu().numpy(), margin_cpu)

    print(f"{r:<6} {coverage:<10.1f}% {max_z_corr:<12.4f} {max_minus_mean_corr:<14.4f} {margin_corr:<12.4f}")

# ============================================================
# Experiment 1c: Easy/Hard Separation with Low-Rank max_minus_mean
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 1c: Easy/Hard Separation with Low-Rank Approximation")
print("=" * 60)
print("(max_minus_mean の低ランク近似で分離性能を検証)")

print(f"\n{'Rank':<6} {'Coverage':<10} {'Easy Loss':<12} {'Hard Loss':<12} {'Loss Diff':<12} {'Oracle %':<12}")
print("-" * 75)

for r in [5, 10, 20, 32, 64]:
    # Precompute low-rank components
    Vt_r = Vt[:r, :]
    S_r = S[:r]
    U_r = U[:, :r]

    max_minus_mean_approx = torch.zeros(num_tokens, device=device)

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_tokens)

            h_chunk = h[start:end]
            h_proj = h_chunk @ Vt_r.T
            h_scaled = h_proj * S_r
            z_approx = h_scaled @ U_r.T

            max_minus_mean_approx[start:end] = z_approx.max(dim=1).values - z_approx.mean(dim=1)

    # Coverage
    coverage = (S[:r] ** 2).sum().item() / (S ** 2).sum().item() * 100

    # Easy/Hard separation
    metric_cpu = max_minus_mean_approx.cpu().numpy()
    threshold = np.median(metric_cpu)
    easy_mask = metric_cpu >= threshold  # higher = easy (negative corr with loss)

    easy_loss = loss_cpu[easy_mask].mean()
    hard_loss = loss_cpu[~easy_mask].mean()
    diff = hard_loss - easy_loss
    oracle_pct = (diff / oracle_diff) * 100

    print(f"{r:<6} {coverage:<10.1f}% {easy_loss:<12.2f} {hard_loss:<12.2f} {diff:<12.2f} {oracle_pct:<12.1f}%")

print(f"{'Exact':<6} {'100.0':<10}% {metrics['max_minus_mean'][metrics['max_minus_mean'] >= np.median(metrics['max_minus_mean'])].mean():<12.2f}", end="")
exact_easy = max_minus_mean_cpu >= np.median(max_minus_mean_cpu)
print(f" {loss_cpu[exact_easy].mean():<12.2f} {loss_cpu[~exact_easy].mean():<12.2f} {loss_cpu[~exact_easy].mean() - loss_cpu[exact_easy].mean():<12.2f} ", end="")
print(f"{(loss_cpu[~exact_easy].mean() - loss_cpu[exact_easy].mean()) / oracle_diff * 100:<12.1f}%")

# ============================================================
# Experiment 2: Correlation with Loss
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 2: Correlation with Loss")
print("=" * 60)

# Move to CPU for correlation analysis
softmax_conf_cpu = softmax_conf.cpu().numpy()
max_z_cpu = max_z.cpu().numpy()
max_minus_mean_cpu = max_minus_mean.cpu().numpy()
margin_cpu = margin.cpu().numpy()
loss_cpu = loss.cpu().numpy()

metrics = {
    "softmax_conf": softmax_conf_cpu,
    "max_z": max_z_cpu,
    "max_minus_mean": max_minus_mean_cpu,
    "margin": margin_cpu,
}

print(f"\n{'Metric':<20} {'Corr with Loss':<15} {'Direction':<12} {'Mean':<12} {'Std':<12}")
print("-" * 75)

for name, metric in metrics.items():
    corr, _ = pearsonr(metric, loss_cpu)
    direction = "lower=easy" if corr > 0 else "higher=easy"
    print(f"{name:<20} {corr:<15.4f} {direction:<12} {metric.mean():<12.4f} {metric.std():<12.4f}")

# ============================================================
# Experiment 2b: MLP Prediction (GPU)
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 2b: MLP Prediction (h -> metric) [GPU]")
print("=" * 60)

torch.manual_seed(42)

# Train/test split
n_train = int(0.8 * num_tokens)
indices = torch.randperm(num_tokens)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

h_train = h[train_idx]
h_test = h[test_idx]

def train_mlp_gpu(X_train, y_train, X_test, y_test, hidden_dim=128, epochs=30, lr=0.01):
    """Simple 2-layer MLP with GPU acceleration."""
    n_samples, input_dim = X_train.shape

    # Normalize targets
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std

    # Initialize weights
    W1 = torch.randn(input_dim, hidden_dim, device=device) * np.sqrt(2.0 / input_dim)
    b1 = torch.zeros(hidden_dim, device=device)
    W2 = torch.randn(hidden_dim, 1, device=device) * np.sqrt(2.0 / hidden_dim)
    b2 = torch.zeros(1, device=device)

    batch_size = 2048
    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        X_shuffled = X_train[perm]
        y_shuffled = y_train_norm[perm]

        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward
            z1 = X_batch @ W1 + b1
            a1 = F.relu(z1)
            pred = (a1 @ W2 + b2).squeeze()

            # Backward (manual gradient)
            grad = 2 * (pred - y_batch) / len(y_batch)
            dW2 = a1.T @ grad.unsqueeze(1)
            db2 = grad.sum()
            da1 = grad.unsqueeze(1) @ W2.T
            dz1 = da1 * (z1 > 0).float()
            dW1 = X_batch.T @ dz1
            db1 = dz1.sum(dim=0)

            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

    # Test
    with torch.no_grad():
        z1 = X_test @ W1 + b1
        a1 = F.relu(z1)
        pred_norm = (a1 @ W2 + b2).squeeze()
        pred = pred_norm * y_std + y_mean

    return pred.cpu().numpy()

print(f"\nTrain: {n_train}, Test: {num_tokens - n_train}")
print(f"\n{'Metric':<20} {'Corr(pred,actual)':<20} {'Time':<10}")
print("-" * 55)

metrics_gpu = {
    "softmax_conf": softmax_conf,
    "max_z": max_z,
    "max_minus_mean": max_minus_mean,
    "margin": margin,
}

for name, metric in metrics_gpu.items():
    mlp_start = time.time()
    y_train = metric[train_idx]
    y_test_gpu = metric[test_idx]

    pred = train_mlp_gpu(h_train, y_train, h_test, y_test_gpu)
    y_test_cpu = y_test_gpu.cpu().numpy()

    corr, _ = pearsonr(pred, y_test_cpu)
    elapsed = time.time() - mlp_start
    print(f"{name:<20} {corr:<20.4f} {elapsed:<10.1f}s")

# ============================================================
# Experiment 2c: Easy/Hard Separation
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 2c: Easy/Hard Separation Quality")
print("=" * 60)

print("\nUsing median threshold:")
print(f"{'Metric':<20} {'Easy Loss':<12} {'Hard Loss':<12} {'Loss Diff':<12} {'Oracle %':<12}")
print("-" * 70)

oracle_threshold = np.median(loss_cpu)
oracle_easy = loss_cpu <= oracle_threshold
oracle_diff = loss_cpu[~oracle_easy].mean() - loss_cpu[oracle_easy].mean()

for name, metric in metrics.items():
    threshold = np.median(metric)
    corr_with_loss, _ = pearsonr(metric, loss_cpu)

    if corr_with_loss < 0:
        easy_mask = metric >= threshold
    else:
        easy_mask = metric <= threshold

    easy_loss = loss_cpu[easy_mask].mean()
    hard_loss = loss_cpu[~easy_mask].mean()
    diff = hard_loss - easy_loss
    oracle_pct = (diff / oracle_diff) * 100

    print(f"{name:<20} {easy_loss:<12.2f} {hard_loss:<12.2f} {diff:<12.2f} {oracle_pct:<12.1f}%")

print(f"{'Oracle':<20} {loss_cpu[oracle_easy].mean():<12.2f} {loss_cpu[~oracle_easy].mean():<12.2f} {oracle_diff:<12.2f} {'100.0':<12}%")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)

print("""
Key Questions Answered:

1. Low-Rank Approximation:
   - Which rank achieves >0.95 correlation with exact softmax_conf?
   - Is the speedup worth the accuracy loss?

2. Alternative Functions:
   - Which metric correlates best with loss? (for separation)
   - Which metric is easiest to predict from h? (for efficiency)

3. Recommendation:
   - Best metric for Easy/Hard separation?
   - Best approach for efficiency?
""")

print(f"\nTotal execution time: GPU accelerated")
