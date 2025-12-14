"""
Low-Rank Approximation and Alternative Functions Analysis

Experiments:
1. Low-rank approximation accuracy for softmax confidence
2. Comparison of different functions for predictability from hidden states
   - softmax_confidence
   - max(z)
   - max(z) - mean(z)
   - margin (max - second_max)

Optimized for memory efficiency with progress logging.
"""

import numpy as np
from scipy.special import softmax
from scipy.stats import pearsonr
import time

# ============================================================
# Load Data
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

data = np.load("hidden_states_data.npz")
h = data["hidden_states"]  # (num_tokens, dim)
loss = data["per_token_loss"]  # (num_tokens,)
W = data["output_head_W"]  # (vocab_size, dim)

num_tokens, dim = h.shape
vocab_size = W.shape[0]

print(f"Hidden states: {h.shape}")
print(f"Per-token loss: {loss.shape}")
print(f"W matrix: {W.shape}")
print(f"Estimated full logits size: {num_tokens * vocab_size * 4 / 1e9:.1f} GB (too large!)")
print("Using chunked processing...")

# ============================================================
# Compute Confidence Metrics in Chunks
# ============================================================
print(f"\n{'=' * 60}")
print("Computing confidence metrics (chunked)...")
print("=" * 60)

chunk_size = 1000  # Smaller chunks for memory efficiency
num_chunks = (num_tokens + chunk_size - 1) // chunk_size

softmax_conf = np.zeros(num_tokens, dtype=np.float32)
max_z = np.zeros(num_tokens, dtype=np.float32)
max_minus_mean = np.zeros(num_tokens, dtype=np.float32)
margin = np.zeros(num_tokens, dtype=np.float32)

start_time = time.time()
print(f"Processing {num_chunks} chunks (chunk_size={chunk_size})...")

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, num_tokens)

    # z_chunk: (chunk_size, vocab_size)
    z_chunk = h[start:end] @ W.T

    # Softmax confidence (max probability) - numerically stable
    z_max = z_chunk.max(axis=1, keepdims=True)
    exp_z = np.exp(z_chunk - z_max)
    softmax_conf[start:end] = (exp_z.max(axis=1) / exp_z.sum(axis=1)).astype(np.float32)

    # Max logit
    max_z[start:end] = z_chunk.max(axis=1).astype(np.float32)

    # Max - mean
    max_minus_mean[start:end] = (z_chunk.max(axis=1) - z_chunk.mean(axis=1)).astype(np.float32)

    # Margin (max - second_max)
    sorted_z = np.partition(z_chunk, -2, axis=1)
    margin[start:end] = (sorted_z[:, -1] - sorted_z[:, -2]).astype(np.float32)

    # Progress logging
    if (i + 1) % 10 == 0 or i == num_chunks - 1:
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

# SVD of W
print("Computing SVD of W...")
svd_start = time.time()
U, S, Vt = np.linalg.svd(W, full_matrices=False)
print(f"SVD done in {time.time() - svd_start:.1f}s")
print(f"SVD shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")

# Test different ranks
ranks = [1, 2, 5, 10, 20, 32, 64]
print(f"\n{'Rank':<6} {'Coverage':<10} {'Corr':<12} {'MSE':<15} {'Time':<10}")
print("-" * 60)

for r in ranks:
    rank_start = time.time()

    # Precompute low-rank components
    Vt_r = Vt[:r, :]  # (r, dim)
    S_r = S[:r]  # (r,)
    U_r = U[:, :r]  # (vocab_size, r)

    softmax_conf_approx = np.zeros(num_tokens, dtype=np.float32)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_tokens)

        h_chunk = h[start:end]  # (chunk, dim)

        # z_approx = h @ Vt_r.T @ diag(S_r) @ U_r.T
        h_proj = h_chunk @ Vt_r.T  # (chunk, r)
        h_scaled = h_proj * S_r  # (chunk, r)
        z_approx = h_scaled @ U_r.T  # (chunk, vocab_size)

        # Softmax confidence
        z_max = z_approx.max(axis=1, keepdims=True)
        exp_z = np.exp(z_approx - z_max)
        softmax_conf_approx[start:end] = (exp_z.max(axis=1) / exp_z.sum(axis=1)).astype(np.float32)

    # Coverage
    coverage = np.sum(S[:r] ** 2) / np.sum(S ** 2) * 100

    # Correlation with exact
    corr, _ = pearsonr(softmax_conf_approx, softmax_conf)

    # MSE
    mse = np.mean((softmax_conf_approx - softmax_conf) ** 2)

    elapsed = time.time() - rank_start
    print(f"{r:<6} {coverage:<10.1f}% {corr:<12.4f} {mse:<15.6f} {elapsed:<10.1f}s")

# ============================================================
# Experiment 2: Correlation with Loss
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 2: Correlation with Loss")
print("=" * 60)

metrics = {
    "softmax_conf": softmax_conf,
    "max_z": max_z,
    "max_minus_mean": max_minus_mean,
    "margin": margin,
}

print(f"\n{'Metric':<20} {'Corr with Loss':<15} {'Direction':<12} {'Mean':<12} {'Std':<12}")
print("-" * 75)

for name, metric in metrics.items():
    corr, _ = pearsonr(metric, loss)
    direction = "lower=easy" if corr > 0 else "higher=easy"
    print(f"{name:<20} {corr:<15.4f} {direction:<12} {metric.mean():<12.4f} {metric.std():<12.4f}")

# ============================================================
# Experiment 2b: MLP Prediction
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 2b: MLP Prediction (h -> metric)")
print("=" * 60)

from numpy.random import default_rng
rng = default_rng(42)

# Train/test split
n_train = int(0.8 * num_tokens)
indices = rng.permutation(num_tokens)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

h_train, h_test = h[train_idx], h[test_idx]

def train_mlp_fast(X_train, y_train, X_test, y_test, hidden_dim=128, epochs=30, lr=0.01):
    """Simple 2-layer MLP with mini-batch SGD."""
    n_samples, input_dim = X_train.shape

    # Normalize
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std

    # Xavier init
    W1 = rng.standard_normal((input_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = rng.standard_normal((hidden_dim, 1)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(1, dtype=np.float32)

    batch_size = 1024
    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train_norm[perm]

        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward
            z1 = X_batch @ W1 + b1
            a1 = np.maximum(0, z1)
            pred = (a1 @ W2 + b2).squeeze()

            # Backward
            grad = 2 * (pred - y_batch) / len(y_batch)
            dW2 = a1.T @ grad.reshape(-1, 1)
            db2 = grad.sum()
            da1 = grad.reshape(-1, 1) @ W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = X_batch.T @ dz1
            db1 = dz1.sum(axis=0)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

    # Test
    z1 = X_test @ W1 + b1
    a1 = np.maximum(0, z1)
    pred_norm = (a1 @ W2 + b2).squeeze()
    pred = pred_norm * y_std + y_mean
    return pred

print(f"\nTrain: {n_train}, Test: {num_tokens - n_train}")
print(f"\n{'Metric':<20} {'Corr(pred,actual)':<20} {'Time':<10}")
print("-" * 55)

for name, metric in metrics.items():
    mlp_start = time.time()
    y_train = metric[train_idx]
    y_test = metric[test_idx]

    pred = train_mlp_fast(h_train, y_train, h_test, y_test)
    corr, _ = pearsonr(pred, y_test)
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

oracle_threshold = np.median(loss)
oracle_easy = loss <= oracle_threshold
oracle_diff = loss[~oracle_easy].mean() - loss[oracle_easy].mean()

for name, metric in metrics.items():
    threshold = np.median(metric)
    corr_with_loss, _ = pearsonr(metric, loss)

    if corr_with_loss < 0:
        easy_mask = metric >= threshold
    else:
        easy_mask = metric <= threshold

    easy_loss = loss[easy_mask].mean()
    hard_loss = loss[~easy_mask].mean()
    diff = hard_loss - easy_loss
    oracle_pct = (diff / oracle_diff) * 100

    print(f"{name:<20} {easy_loss:<12.2f} {hard_loss:<12.2f} {diff:<12.2f} {oracle_pct:<12.1f}%")

print(f"{'Oracle':<20} {loss[oracle_easy].mean():<12.2f} {loss[~oracle_easy].mean():<12.2f} {oracle_diff:<12.2f} {'100.0':<12}%")

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
