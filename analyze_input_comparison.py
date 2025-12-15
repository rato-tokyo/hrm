"""
Analyze Different Input Features for Saturation Prediction

比較する入力:
1. h_out: 層の出力（累積 + この層の変化）
2. delta: この層の変化のみ
3. h_out + delta: 両方を結合
4. delta + prev_delta: この層と前の層の変化を結合

Usage:
    python analyze_input_comparison.py
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


class MLPPredictor(nn.Module):
    """MLP predictor for saturation."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


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
) -> dict:
    """Train MLP and return metrics."""

    model = MLPPredictor(input_dim, hidden_dim).to(DEVICE)
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

    # Load best model and final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_preds = (torch.sigmoid(val_logits) > 0.5).float()

        labels_np = y_val.cpu().numpy()
        preds_np = val_preds.cpu().numpy()

        return {
            'f1': f1_score(labels_np, preds_np),
            'precision': precision_score(labels_np, preds_np, zero_division=0),
            'recall': recall_score(labels_np, preds_np, zero_division=0),
            'pred_positive_rate': preds_np.mean(),
        }


def experiment_layer(
    data,
    W: torch.Tensor,
    top1_final: torch.Tensor,
    layer: int,
    dim: int,
    prev_delta: torch.Tensor = None,
):
    """Run experiment for a specific layer."""
    print(f"\n{'=' * 60}")
    print(f"Layer {layer}")
    print("=" * 60)

    # Load data
    h_out = torch.from_numpy(data[f'layer{layer}_h_out']).float().to(DEVICE)
    delta = torch.from_numpy(data[f'layer{layer}_delta']).float().to(DEVICE)

    # Compute top-1 at this layer
    top1_here = compute_top1_batched(h_out, W)

    # Labels: 1 = can exit (top-1 matches final), 0 = cannot exit
    labels = (top1_here == top1_final).float()
    saturation_rate = labels.mean().item()
    print(f"Saturation rate: {saturation_rate * 100:.1f}%")

    # Train/val split
    num_samples = h_out.shape[0]
    num_train = int(num_samples * 0.8)
    perm = torch.randperm(num_samples, device=DEVICE)
    train_idx, val_idx = perm[:num_train], perm[num_train:]

    hidden_dim = dim
    results = {}

    # 1. h_out only
    print("\n--- Method 1: h_out only ---")
    r1 = train_and_evaluate(
        h_out[train_idx], labels[train_idx],
        h_out[val_idx], labels[val_idx],
        input_dim=dim, hidden_dim=hidden_dim,
    )
    print(f"F1: {r1['f1'] * 100:.1f}%")
    results['h_out'] = r1

    # 2. delta only
    print("\n--- Method 2: delta only ---")
    r2 = train_and_evaluate(
        delta[train_idx], labels[train_idx],
        delta[val_idx], labels[val_idx],
        input_dim=dim, hidden_dim=hidden_dim,
    )
    print(f"F1: {r2['f1'] * 100:.1f}%")
    results['delta'] = r2

    # 3. h_out + delta concatenated
    print("\n--- Method 3: h_out + delta ---")
    h_out_delta = torch.cat([h_out, delta], dim=-1)
    r3 = train_and_evaluate(
        h_out_delta[train_idx], labels[train_idx],
        h_out_delta[val_idx], labels[val_idx],
        input_dim=dim * 2, hidden_dim=hidden_dim,
    )
    print(f"F1: {r3['f1'] * 100:.1f}%")
    results['h_out_delta'] = r3

    # 4. delta + prev_delta (if available)
    if prev_delta is not None:
        print("\n--- Method 4: delta + prev_delta ---")
        delta_prev = torch.cat([delta, prev_delta], dim=-1)
        r4 = train_and_evaluate(
            delta_prev[train_idx], labels[train_idx],
            delta_prev[val_idx], labels[val_idx],
            input_dim=dim * 2, hidden_dim=hidden_dim,
        )
        print(f"F1: {r4['f1'] * 100:.1f}%")
        results['delta_prev'] = r4
    else:
        print("\n--- Method 4: delta + prev_delta (N/A for layer 1) ---")
        results['delta_prev'] = None

    results['saturation_rate'] = saturation_rate

    return results, delta


def main():
    data_path = "layerwise_residuals.npz"
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found.")
        return

    data, dim, vocab_size, num_layers = load_data(data_path)
    W = torch.from_numpy(data['output_head_W']).to(DEVICE)

    print(f"\nNumber of tokens: {data['layer1_h_out'].shape[0]}")

    # Compute final layer top-1
    print("\nComputing final layer top-1...")
    h_final = torch.from_numpy(data[f'layer{num_layers}_h_out']).to(DEVICE)
    top1_final = compute_top1_batched(h_final, W)

    # Run experiments
    all_results = {}
    prev_delta = None

    for layer in range(1, num_layers):
        results, current_delta = experiment_layer(
            data, W, top1_final, layer, dim, prev_delta
        )
        all_results[layer] = results
        prev_delta = current_delta

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Layer':<8} {'Sat%':<8} {'h_out':<10} {'delta':<10} {'h+d':<10} {'d+prev':<10}")
    print("-" * 56)

    h_out_f1s = []
    delta_f1s = []
    h_out_delta_f1s = []
    delta_prev_f1s = []

    for layer, results in all_results.items():
        sat = results['saturation_rate'] * 100
        h_out_f1 = results['h_out']['f1'] * 100
        delta_f1 = results['delta']['f1'] * 100
        h_out_delta_f1 = results['h_out_delta']['f1'] * 100

        h_out_f1s.append(h_out_f1)
        delta_f1s.append(delta_f1)
        h_out_delta_f1s.append(h_out_delta_f1)

        if results['delta_prev'] is not None:
            delta_prev_f1 = results['delta_prev']['f1'] * 100
            delta_prev_f1s.append(delta_prev_f1)
            print(f"{layer:<8} {sat:<7.1f}% {h_out_f1:<9.1f}% {delta_f1:<9.1f}% {h_out_delta_f1:<9.1f}% {delta_prev_f1:<9.1f}%")
        else:
            print(f"{layer:<8} {sat:<7.1f}% {h_out_f1:<9.1f}% {delta_f1:<9.1f}% {h_out_delta_f1:<9.1f}% {'N/A':<10}")

    print("-" * 56)

    avg_h_out = np.mean(h_out_f1s)
    avg_delta = np.mean(delta_f1s)
    avg_h_out_delta = np.mean(h_out_delta_f1s)
    avg_delta_prev = np.mean(delta_prev_f1s) if delta_prev_f1s else 0

    print(f"{'Average':<8} {'':<8} {avg_h_out:<9.1f}% {avg_delta:<9.1f}% {avg_h_out_delta:<9.1f}% {avg_delta_prev:<9.1f}%")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    print(f"\n1. h_out vs delta:")
    if avg_h_out > avg_delta:
        print(f"   h_out が優位（+{avg_h_out - avg_delta:.1f}%）")
        print(f"   → 累積情報が saturation 予測に重要")
    else:
        print(f"   delta が優位（+{avg_delta - avg_h_out:.1f}%）")
        print(f"   → この層の変化が saturation 予測に重要")

    print(f"\n2. h_out + delta の効果:")
    improvement = avg_h_out_delta - max(avg_h_out, avg_delta)
    if improvement > 0:
        print(f"   結合で改善（+{improvement:.1f}%）")
        print(f"   → 両方の情報が相補的")
    else:
        print(f"   結合による改善なし（{improvement:.1f}%）")
        print(f"   → 情報が冗長")

    if delta_prev_f1s:
        print(f"\n3. delta + prev_delta の効果:")
        improvement_prev = avg_delta_prev - avg_delta
        if improvement_prev > 0:
            print(f"   前層の delta で改善（+{improvement_prev:.1f}%）")
            print(f"   → 変化の履歴が有用")
        else:
            print(f"   前層の delta による改善なし（{improvement_prev:.1f}%）")

    # Best method
    methods = {
        'h_out': avg_h_out,
        'delta': avg_delta,
        'h_out + delta': avg_h_out_delta,
    }
    if delta_prev_f1s:
        methods['delta + prev_delta'] = avg_delta_prev

    best_method = max(methods, key=methods.get)
    print(f"\n結論: 最良の入力は「{best_method}」（平均 F1 = {methods[best_method]:.1f}%）")


if __name__ == "__main__":
    main()
