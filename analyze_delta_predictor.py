"""
Analyze Delta-based Saturation Prediction

delta（残差ベクトル）全体を使ってsaturationを予測できるか検証。

比較:
1. Linear: delta → Linear → sigmoid → saturation予測
2. MLP: delta → Linear → ReLU → Linear → sigmoid → saturation予測

既存の layerwise_residuals.npz を使用。

Usage:
    python analyze_delta_predictor.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
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


class LinearPredictor(nn.Module):
    """Linear predictor: delta -> saturation probability."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.fc(delta).squeeze(-1)


class MLPPredictor(nn.Module):
    """MLP predictor: delta -> saturation probability."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(delta))
        return self.fc2(h).squeeze(-1)


def train_predictor(
    model: nn.Module,
    delta_train: torch.Tensor,
    labels_train: torch.Tensor,
    delta_val: torch.Tensor,
    labels_val: torch.Tensor,
    lr: float,
    num_epochs: int,
    batch_size: int,
) -> dict:
    """Train a predictor model."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Move data to device
    delta_train = delta_train.to(DEVICE)
    labels_train = labels_train.to(DEVICE)
    delta_val = delta_val.to(DEVICE)
    labels_val = labels_val.to(DEVICE)

    num_samples = delta_train.shape[0]
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Shuffle
        perm = torch.randperm(num_samples, device=DEVICE)
        delta_train = delta_train[perm]
        labels_train = labels_train[perm]

        for i in range(0, num_samples, batch_size):
            delta_batch = delta_train[i:i + batch_size]
            labels_batch = labels_train[i:i + batch_size]

            logits = model(delta_batch)
            loss = F.binary_cross_entropy_with_logits(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * delta_batch.shape[0]

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(delta_val)
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()

            val_f1 = f1_score(labels_val.cpu().numpy(), val_preds.cpu().numpy())

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(delta_val)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).float()

        labels_np = labels_val.cpu().numpy()
        preds_np = val_preds.cpu().numpy()

        f1 = f1_score(labels_np, preds_np)
        precision = precision_score(labels_np, preds_np)
        recall = recall_score(labels_np, preds_np)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def experiment_layer(
    data,
    W: torch.Tensor,
    layer_current: int,
    layer_next: int,
    dim: int,
):
    """Run experiment for a specific layer transition."""
    print(f"\n{'=' * 60}")
    print(f"Layer {layer_current} → Layer {layer_next}")
    print("=" * 60)

    # Load data
    h_out_current = torch.from_numpy(data[f'layer{layer_current}_h_out']).to(DEVICE)
    h_out_next = torch.from_numpy(data[f'layer{layer_next}_h_out']).to(DEVICE)
    delta = torch.from_numpy(data[f'layer{layer_next}_delta']).float().to(DEVICE)

    # Compute top-1
    print("Computing top-1 predictions...")
    top1_current = compute_top1_batched(h_out_current, W)
    top1_next = compute_top1_batched(h_out_next, W)

    # Labels: 1 = saturated (top-1 didn't change), 0 = changed
    labels = (top1_current == top1_next).float()
    saturation_rate = labels.mean().item()
    print(f"Saturation rate: {saturation_rate * 100:.1f}%")

    # Split train/val (80/20)
    num_samples = delta.shape[0]
    num_train = int(num_samples * 0.8)

    perm = torch.randperm(num_samples, device=DEVICE)
    train_idx = perm[:num_train]
    val_idx = perm[num_train:]

    delta_train = delta[train_idx]
    labels_train = labels[train_idx]
    delta_val = delta[val_idx]
    labels_val = labels[val_idx]

    print(f"Train: {num_train}, Val: {num_samples - num_train}")

    # Training config
    lr = 1e-3
    num_epochs = 20
    batch_size = 1024
    hidden_dim = dim  # Same as input dim

    results = {}

    # 1. Linear predictor
    print("\n--- Linear Predictor ---")
    linear_model = LinearPredictor(dim)
    linear_results = train_predictor(
        linear_model, delta_train, labels_train, delta_val, labels_val,
        lr, num_epochs, batch_size
    )
    print(f"F1: {linear_results['f1'] * 100:.1f}%")
    print(f"Precision: {linear_results['precision'] * 100:.1f}%")
    print(f"Recall: {linear_results['recall'] * 100:.1f}%")
    results['linear'] = linear_results

    # 2. MLP predictor
    print("\n--- MLP Predictor ---")
    mlp_model = MLPPredictor(dim, hidden_dim)
    mlp_results = train_predictor(
        mlp_model, delta_train, labels_train, delta_val, labels_val,
        lr, num_epochs, batch_size
    )
    print(f"F1: {mlp_results['f1'] * 100:.1f}%")
    print(f"Precision: {mlp_results['precision'] * 100:.1f}%")
    print(f"Recall: {mlp_results['recall'] * 100:.1f}%")
    results['mlp'] = mlp_results

    # 3. Baseline: delta_norm threshold (from previous experiment)
    print("\n--- Baseline: Delta Norm Threshold ---")
    delta_norm = torch.norm(delta_val, dim=-1)

    best_f1 = 0
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        threshold = torch.quantile(delta_norm, percentile / 100.0)
        # Low delta_norm = saturated
        preds = (delta_norm < threshold).float()

        f1 = f1_score(labels_val.cpu().numpy(), preds.cpu().numpy())
        if f1 > best_f1:
            best_f1 = f1

    print(f"Best F1: {best_f1 * 100:.1f}%")
    results['delta_norm'] = {'f1': best_f1}

    return results


def main():
    # Check if data file exists
    data_path = "layerwise_residuals.npz"
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found.")
        print("Please run save_layerwise_residuals.py first.")
        return

    # Load data
    data, dim, vocab_size, num_layers = load_data(data_path)
    W = torch.from_numpy(data['output_head_W']).to(DEVICE)

    print(f"\nOutput head W: {W.shape}")
    print(f"Number of tokens: {data['layer1_h_out'].shape[0]}")

    # Run experiments for each layer transition
    all_results = {}

    for i in range(num_layers - 1):
        layer_current = i + 1
        layer_next = i + 2

        results = experiment_layer(data, W, layer_current, layer_next, dim)
        all_results[f"{layer_current}→{layer_next}"] = results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n{:<12} {:>12} {:>12} {:>12}".format(
        "Transition", "Delta Norm", "Linear", "MLP"
    ))
    print("-" * 50)

    linear_f1s = []
    mlp_f1s = []
    norm_f1s = []

    for transition, results in all_results.items():
        norm_f1 = results['delta_norm']['f1'] * 100
        linear_f1 = results['linear']['f1'] * 100
        mlp_f1 = results['mlp']['f1'] * 100

        norm_f1s.append(norm_f1)
        linear_f1s.append(linear_f1)
        mlp_f1s.append(mlp_f1)

        print(f"{transition:<12} {norm_f1:>11.1f}% {linear_f1:>11.1f}% {mlp_f1:>11.1f}%")

    print("-" * 50)
    print(f"{'Average':<12} {np.mean(norm_f1s):>11.1f}% {np.mean(linear_f1s):>11.1f}% {np.mean(mlp_f1s):>11.1f}%")

    print("\n結論:")
    avg_norm = np.mean(norm_f1s)
    avg_linear = np.mean(linear_f1s)
    avg_mlp = np.mean(mlp_f1s)

    best_method = "MLP" if avg_mlp >= avg_linear and avg_mlp >= avg_norm else \
                  "Linear" if avg_linear >= avg_norm else "Delta Norm"

    print(f"  - Delta Norm (threshold): 平均 F1 = {avg_norm:.1f}%")
    print(f"  - Linear (学習):          平均 F1 = {avg_linear:.1f}%")
    print(f"  - MLP (学習):             平均 F1 = {avg_mlp:.1f}%")
    print(f"  → 最良: {best_method}")

    if avg_mlp > avg_norm + 5:
        print(f"  → deltaベクトル全体を使うことで +{avg_mlp - avg_norm:.1f}% の改善")
    elif avg_mlp > avg_norm:
        print(f"  → 小幅な改善 (+{avg_mlp - avg_norm:.1f}%)")
    else:
        print(f"  → deltaベクトル全体を使っても改善なし")


if __name__ == "__main__":
    main()
