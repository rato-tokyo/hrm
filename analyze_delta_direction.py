"""
Analyze Delta Direction and Saturation

仮説: delta の方向が top-1 の logit を強める方向なら saturation
      delta の方向が top-1 を変える方向なら非 saturation

実験:
1. delta と top-1 token の W ベクトルの cosine similarity
2. delta による top-1 logit の変化量
3. MLP で delta の方向情報を使った saturation 予測

Usage:
    python analyze_delta_direction.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.metrics import f1_score, precision_score, recall_score


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def load_data(path: str = "layerwise_residuals.npz"):
    """Load residual data."""
    print(f"Loading {path}...")
    data = np.load(path, allow_pickle=True)

    dim = int(data['dim'])
    vocab_size = int(data['vocab_size'])
    num_layers = int(data['num_layers'])

    print(f"Config: dim={dim}, vocab_size={vocab_size}, num_layers={num_layers}")
    print(f"Device: {DEVICE}")

    return data, dim, vocab_size, num_layers


def compute_top1_batched(
    h: torch.Tensor,
    W: torch.Tensor,
    batch_size: int = 8192
) -> torch.Tensor:
    """Compute top-1 predictions in batches using GPU."""
    num_tokens = h.shape[0]
    top1_list = []

    with torch.no_grad():
        for i in range(0, num_tokens, batch_size):
            h_batch = h[i:i + batch_size]
            logits_batch = h_batch @ W.T
            top1_batch = logits_batch.argmax(dim=-1)
            top1_list.append(top1_batch)

    return torch.cat(top1_list)


def analyze_delta_direction(data, W, num_layers, dim):
    """Analyze delta direction relative to top-1 token."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Delta Direction vs Top-1 W Vector")
    print("=" * 60)

    print("\nFor each token:")
    print("  - Get top-1 token at current layer")
    print("  - Compute cosine similarity between delta and W[top-1]")
    print("  - Check if high similarity correlates with saturation")

    # Get final layer top-1 for saturation labels
    h_final = torch.from_numpy(data[f'layer{num_layers}_h_out']).to(DEVICE)
    top1_final = compute_top1_batched(h_final, W)

    results = []

    for layer in range(1, num_layers):
        print(f"\n--- Layer {layer} ---")

        # Load data
        h_out = torch.from_numpy(data[f'layer{layer}_h_out']).to(DEVICE)
        delta = torch.from_numpy(data[f'layer{layer}_delta']).to(DEVICE)

        # Get top-1 at this layer
        top1_here = compute_top1_batched(h_out, W)

        # Get W vectors for top-1 tokens
        W_top1 = W[top1_here]  # (num_tokens, dim)

        # Normalize for cosine similarity
        delta_norm = F.normalize(delta, dim=-1)
        W_top1_norm = F.normalize(W_top1, dim=-1)

        # Cosine similarity between delta and W[top-1]
        cos_sim = (delta_norm * W_top1_norm).sum(dim=-1)  # (num_tokens,)

        # Is this token saturated? (top-1 matches final)
        is_saturated = (top1_here == top1_final).float()
        saturation_rate = is_saturated.mean().item()

        # Correlation
        cos_sim_np = cos_sim.cpu().numpy()
        is_saturated_np = is_saturated.cpu().numpy()

        corr, p_value = scipy_stats.pointbiserialr(cos_sim_np, is_saturated_np)

        # Statistics for saturated vs non-saturated
        saturated_mask = is_saturated_np == 1
        cos_sim_saturated = cos_sim_np[saturated_mask].mean()
        cos_sim_not_saturated = cos_sim_np[~saturated_mask].mean()

        print(f"Saturation rate: {saturation_rate * 100:.1f}%")
        print(f"Cosine sim (delta, W[top-1]): mean={cos_sim_np.mean():.4f}")
        print(f"  Saturated tokens: {cos_sim_saturated:.4f}")
        print(f"  Non-saturated tokens: {cos_sim_not_saturated:.4f}")
        print(f"Correlation: r={corr:.3f} (p={p_value:.2e})")

        results.append({
            'layer': layer,
            'saturation_rate': saturation_rate,
            'cos_sim_mean': cos_sim_np.mean(),
            'cos_sim_saturated': cos_sim_saturated,
            'cos_sim_not_saturated': cos_sim_not_saturated,
            'correlation': corr,
        })

    return results


def analyze_logit_change(data, W, num_layers, dim):
    """Analyze how delta changes the top-1 logit."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Delta's Effect on Top-1 Logit")
    print("=" * 60)

    print("\nFor each token:")
    print("  - Compute logit change: delta @ W[top-1].T")
    print("  - Positive = delta strengthens top-1")
    print("  - Negative = delta weakens top-1")

    h_final = torch.from_numpy(data[f'layer{num_layers}_h_out']).to(DEVICE)
    top1_final = compute_top1_batched(h_final, W)

    results = []

    for layer in range(1, num_layers):
        print(f"\n--- Layer {layer} ---")

        h_out = torch.from_numpy(data[f'layer{layer}_h_out']).to(DEVICE)
        delta = torch.from_numpy(data[f'layer{layer}_delta']).to(DEVICE)

        top1_here = compute_top1_batched(h_out, W)

        # Get W vectors for top-1 tokens
        W_top1 = W[top1_here]  # (num_tokens, dim)

        # Logit change for top-1: delta @ W[top-1]
        logit_change = (delta * W_top1).sum(dim=-1)  # (num_tokens,)

        # Is saturated?
        is_saturated = (top1_here == top1_final).float()

        logit_change_np = logit_change.cpu().numpy()
        is_saturated_np = is_saturated.cpu().numpy()

        # Correlation
        corr, p_value = scipy_stats.pointbiserialr(logit_change_np, is_saturated_np)

        # Statistics
        saturated_mask = is_saturated_np == 1
        change_saturated = logit_change_np[saturated_mask].mean()
        change_not_saturated = logit_change_np[~saturated_mask].mean()

        # Positive rate (delta strengthens top-1)
        positive_rate = (logit_change_np > 0).mean()

        print(f"Logit change: mean={logit_change_np.mean():.4f}")
        print(f"  Saturated tokens: {change_saturated:.4f}")
        print(f"  Non-saturated tokens: {change_not_saturated:.4f}")
        print(f"Positive rate (delta strengthens top-1): {positive_rate * 100:.1f}%")
        print(f"Correlation: r={corr:.3f} (p={p_value:.2e})")

        results.append({
            'layer': layer,
            'logit_change_saturated': change_saturated,
            'logit_change_not_saturated': change_not_saturated,
            'positive_rate': positive_rate,
            'correlation': corr,
        })

    return results


def analyze_direction_features(data, W, num_layers, dim):
    """Use direction-based features to predict saturation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Direction Features for Saturation Prediction")
    print("=" * 60)

    print("\nFeatures:")
    print("  1. cos_sim(delta, W[top-1])")
    print("  2. logit_change = delta @ W[top-1]")
    print("  3. delta_norm")
    print("\nCompare: direction features vs delta vector vs combined")

    h_final = torch.from_numpy(data[f'layer{num_layers}_h_out']).to(DEVICE)
    top1_final = compute_top1_batched(h_final, W)

    results = []

    for layer in range(1, num_layers):
        print(f"\n--- Layer {layer} ---")

        h_out = torch.from_numpy(data[f'layer{layer}_h_out']).to(DEVICE)
        delta = torch.from_numpy(data[f'layer{layer}_delta']).to(DEVICE)

        top1_here = compute_top1_batched(h_out, W)
        W_top1 = W[top1_here]

        # Compute features
        delta_norm_vec = delta.norm(dim=-1, keepdim=True)
        delta_normalized = F.normalize(delta, dim=-1)
        W_top1_normalized = F.normalize(W_top1, dim=-1)

        cos_sim = (delta_normalized * W_top1_normalized).sum(dim=-1, keepdim=True)
        logit_change = (delta * W_top1).sum(dim=-1, keepdim=True)

        # Direction features: [cos_sim, logit_change, delta_norm]
        direction_features = torch.cat([cos_sim, logit_change, delta_norm_vec], dim=-1)

        # Labels
        is_saturated = (top1_here == top1_final).float()

        # Train/val split
        num_samples = delta.shape[0]
        num_train = int(num_samples * 0.8)
        perm = torch.randperm(num_samples, device=DEVICE)
        train_idx, val_idx = perm[:num_train], perm[num_train:]

        # Method 1: Direction features only (3 features)
        f1_direction = train_and_evaluate(
            direction_features[train_idx],
            is_saturated[train_idx],
            direction_features[val_idx],
            is_saturated[val_idx],
            input_dim=3,
            hidden_dim=16,
        )

        # Method 2: Delta vector (dim features)
        f1_delta = train_and_evaluate(
            delta[train_idx],
            is_saturated[train_idx],
            delta[val_idx],
            is_saturated[val_idx],
            input_dim=dim,
            hidden_dim=dim,
        )

        # Method 3: Combined [direction_features, delta]
        combined = torch.cat([direction_features, delta], dim=-1)
        f1_combined = train_and_evaluate(
            combined[train_idx],
            is_saturated[train_idx],
            combined[val_idx],
            is_saturated[val_idx],
            input_dim=3 + dim,
            hidden_dim=dim,
        )

        print(f"F1 (direction features only, 3 dims): {f1_direction * 100:.1f}%")
        print(f"F1 (delta vector, {dim} dims):        {f1_delta * 100:.1f}%")
        print(f"F1 (combined, {3 + dim} dims):          {f1_combined * 100:.1f}%")

        results.append({
            'layer': layer,
            'f1_direction': f1_direction,
            'f1_delta': f1_delta,
            'f1_combined': f1_combined,
        })

    return results


def train_and_evaluate(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    hidden_dim: int,
    lr: float = 1e-3,
    num_epochs: int = 20,
    batch_size: int = 1024,
) -> float:
    """Train MLP and return F1 score."""

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x))).squeeze(-1)

    model = MLP(input_dim, hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(X_train.shape[0], device=DEVICE)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            logits = model(X_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            f1 = f1_score(y_val.cpu().numpy(), val_preds.cpu().numpy())

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_f1


def main():
    data_path = "layerwise_residuals.npz"
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found.")
        return

    data, dim, vocab_size, num_layers = load_data(data_path)
    W = torch.from_numpy(data['output_head_W']).to(DEVICE)

    print(f"\nNumber of tokens: {data['layer1_h_out'].shape[0]}")

    # Run experiments
    exp1_results = analyze_delta_direction(data, W, num_layers, dim)
    exp2_results = analyze_logit_change(data, W, num_layers, dim)
    exp3_results = analyze_direction_features(data, W, num_layers, dim)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n[Experiment 1] Cosine Similarity (delta, W[top-1]) vs Saturation:")
    for r in exp1_results:
        print(f"  Layer {r['layer']}: r={r['correlation']:.3f}, "
              f"sat={r['cos_sim_saturated']:.4f}, non-sat={r['cos_sim_not_saturated']:.4f}")

    print("\n[Experiment 2] Logit Change vs Saturation:")
    for r in exp2_results:
        print(f"  Layer {r['layer']}: r={r['correlation']:.3f}, "
              f"sat={r['logit_change_saturated']:.4f}, non-sat={r['logit_change_not_saturated']:.4f}")

    print("\n[Experiment 3] F1 Comparison:")
    print(f"{'Layer':<8} {'Direction':<12} {'Delta':<12} {'Combined':<12}")
    print("-" * 44)
    for r in exp3_results:
        print(f"{r['layer']:<8} {r['f1_direction'] * 100:<11.1f}% {r['f1_delta'] * 100:<11.1f}% {r['f1_combined'] * 100:<11.1f}%")

    avg_direction = np.mean([r['f1_direction'] for r in exp3_results])
    avg_delta = np.mean([r['f1_delta'] for r in exp3_results])
    avg_combined = np.mean([r['f1_combined'] for r in exp3_results])

    print("-" * 44)
    print(f"{'Average':<8} {avg_direction * 100:<11.1f}% {avg_delta * 100:<11.1f}% {avg_combined * 100:<11.1f}%")

    print("\n結論:")
    if avg_direction > avg_delta:
        print(f"  → 方向特徴（3次元）が delta ベクトル（{dim}次元）より効果的！")
    elif avg_combined > avg_delta:
        print(f"  → 方向特徴を追加すると改善（{avg_combined * 100:.1f}% vs {avg_delta * 100:.1f}%）")
    else:
        print(f"  → delta ベクトルが最も効果的（{avg_delta * 100:.1f}%）")


if __name__ == "__main__":
    main()
