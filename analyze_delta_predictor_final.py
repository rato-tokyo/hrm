"""
Analyze Delta-based Final Saturation Prediction

delta（残差ベクトル）を使って「現在の層でexitして良いか」を予測。

予測対象: Layer N の top-1 が最終層の top-1 と一致するか
入力: Layer N の delta

これは実用的なexit判定に直接使える。

Usage:
    python analyze_delta_predictor_final.py
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
    """Linear predictor: delta -> exit probability."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.fc(delta).squeeze(-1)


class MLPPredictor(nn.Module):
    """MLP predictor: delta -> exit probability."""

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
        delta_train_shuffled = delta_train[perm]
        labels_train_shuffled = labels_train[perm]

        for i in range(0, num_samples, batch_size):
            delta_batch = delta_train_shuffled[i:i + batch_size]
            labels_batch = labels_train_shuffled[i:i + batch_size]

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
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)

        # Debug: prediction distribution
        pred_positive_rate = preds_np.mean()
        label_positive_rate = labels_np.mean()

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pred_positive_rate': pred_positive_rate,
        'label_positive_rate': label_positive_rate,
    }


def experiment_layer(
    data,
    W: torch.Tensor,
    top1_final: torch.Tensor,
    layer: int,
    dim: int,
):
    """Run experiment for a specific layer."""
    print(f"\n{'=' * 60}")
    print(f"Layer {layer}: Can we exit here?")
    print("=" * 60)

    # Load data
    h_out = torch.from_numpy(data[f'layer{layer}_h_out']).to(DEVICE)
    delta = torch.from_numpy(data[f'layer{layer}_delta']).float().to(DEVICE)

    # Compute top-1 at this layer
    print("Computing top-1 predictions...")
    top1_here = compute_top1_batched(h_out, W)

    # Labels: 1 = can exit (top-1 matches final), 0 = cannot exit
    labels = (top1_here == top1_final).float()
    exit_rate = labels.mean().item()
    print(f"Exit rate (matches final): {exit_rate * 100:.1f}%")

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
    hidden_dim = dim

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
    print(f"Pred positive rate: {mlp_results['pred_positive_rate'] * 100:.1f}% (label: {mlp_results['label_positive_rate'] * 100:.1f}%)")
    results['mlp'] = mlp_results

    # 3. Baseline: delta_norm threshold
    print("\n--- Baseline: Delta Norm Threshold ---")
    delta_norm = torch.norm(delta_val, dim=-1)

    best_f1 = 0
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        threshold = torch.quantile(delta_norm, percentile / 100.0)
        # Low delta_norm = can exit (assumption)
        preds = (delta_norm < threshold).float()

        f1 = f1_score(labels_val.cpu().numpy(), preds.cpu().numpy())
        if f1 > best_f1:
            best_f1 = f1

    print(f"Best F1: {best_f1 * 100:.1f}%")
    results['delta_norm'] = {'f1': best_f1}

    results['exit_rate'] = exit_rate

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

    # Compute final layer top-1 (ground truth)
    print("\nComputing final layer top-1 (ground truth)...")
    h_final = torch.from_numpy(data[f'layer{num_layers}_h_out']).to(DEVICE)
    top1_final = compute_top1_batched(h_final, W)
    print(f"Final layer top-1 computed.")

    # Run experiments for each layer (except final)
    all_results = {}

    for layer in range(1, num_layers):  # Layer 1 to num_layers-1
        results = experiment_layer(data, W, top1_final, layer, dim)
        all_results[f"Layer {layer}"] = results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Can we exit at each layer?")
    print("=" * 60)
    print("\nPrediction target: top-1 at Layer N == top-1 at Final Layer")

    print("\n{:<12} {:>10} {:>12} {:>12} {:>12}".format(
        "Layer", "Exit Rate", "Delta Norm", "Linear", "MLP"
    ))
    print("-" * 60)

    linear_f1s = []
    mlp_f1s = []
    norm_f1s = []

    for layer_name, results in all_results.items():
        exit_rate = results['exit_rate'] * 100
        norm_f1 = results['delta_norm']['f1'] * 100
        linear_f1 = results['linear']['f1'] * 100
        mlp_f1 = results['mlp']['f1'] * 100

        norm_f1s.append(norm_f1)
        linear_f1s.append(linear_f1)
        mlp_f1s.append(mlp_f1)

        print(f"{layer_name:<12} {exit_rate:>9.1f}% {norm_f1:>11.1f}% {linear_f1:>11.1f}% {mlp_f1:>11.1f}%")

    print("-" * 60)
    print(f"{'Average':<12} {'':<10} {np.mean(norm_f1s):>11.1f}% {np.mean(linear_f1s):>11.1f}% {np.mean(mlp_f1s):>11.1f}%")

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

    print("\n解釈:")
    print("  - Exit Rate: その層で exit した場合に最終層と一致する割合")
    print("  - F1: delta から「exit すべきか」を予測する精度")
    print("  - 高い F1 = delta で exit 判定が可能")


if __name__ == "__main__":
    main()
