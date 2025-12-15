"""
Analyze CALM State Propagation Method

CALM の State Propagation を忠実に再現:
- 連続する2層間の hidden states の cosine similarity を使用
- cos_sim(h_layer_n, h_layer_n+1) で exit 判定

CALM 論文の定義:
"State Propagation: the cosine similarity between the hidden states
of consecutive layers"

Usage:
    python analyze_calm_state_propagation.py
"""

import numpy as np
import torch
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


def experiment_calm_state_propagation(data, W, num_layers, top1_final):
    """
    CALM State Propagation: cos_sim(h_layer_n, h_layer_n+1)

    Layer N で exit するかどうかを、
    Layer N と Layer N+1 の hidden states の cosine similarity で判定。

    予測対象: top1_layer_n == top1_final
    """
    print("\n" + "=" * 60)
    print("CALM State Propagation: cos_sim(h_n, h_n+1)")
    print("=" * 60)

    print("\n判定: Layer N と Layer N+1 の cos_sim が高い → Layer N で exit")
    print("予測対象: top1_layer_n == top1_final\n")

    results = []

    for layer in range(1, num_layers):  # Layer 1 to num_layers-1
        print(f"--- Layer {layer} ---")

        # Load h_out for this layer and next layer
        h_n = torch.from_numpy(data[f'layer{layer}_h_out']).float().to(DEVICE)
        h_n_plus_1 = torch.from_numpy(data[f'layer{layer + 1}_h_out']).float().to(DEVICE)

        # Compute cosine similarity between consecutive layers
        h_n_norm = F.normalize(h_n, dim=-1)
        h_n_plus_1_norm = F.normalize(h_n_plus_1, dim=-1)
        cos_sim = (h_n_norm * h_n_plus_1_norm).sum(dim=-1)  # (num_tokens,)

        # Labels: top1 at layer N matches final
        top1_n = compute_top1_batched(h_n, W)
        labels = (top1_n == top1_final).float()
        saturation_rate = labels.mean().item()

        # Train/val split (use val for threshold search)
        num_samples = h_n.shape[0]
        num_train = int(num_samples * 0.8)
        perm = torch.randperm(num_samples, device=DEVICE)
        val_idx = perm[num_train:]

        cos_sim_val = cos_sim[val_idx]
        labels_val = labels[val_idx].cpu().numpy()

        # Find best threshold
        best_f1 = 0
        best_threshold = 0
        best_precision = 0
        best_recall = 0

        for percentile in range(10, 100, 5):  # Finer search
            threshold = torch.quantile(cos_sim_val, percentile / 100.0).item()
            # High cos_sim = small change = likely saturated = can exit
            preds = (cos_sim_val > threshold).float().cpu().numpy()

            f1 = f1_score(labels_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision_score(labels_val, preds, zero_division=0)
                best_recall = recall_score(labels_val, preds, zero_division=0)

        print(f"Saturation rate: {saturation_rate * 100:.1f}%")
        print(f"cos_sim mean: {cos_sim.mean().item():.4f}")
        print(f"F1: {best_f1 * 100:.1f}% (P={best_precision * 100:.1f}%, R={best_recall * 100:.1f}%)")
        print(f"Threshold: {best_threshold:.4f}")

        results.append({
            'layer': layer,
            'saturation_rate': saturation_rate,
            'cos_sim_mean': cos_sim.mean().item(),
            'f1': best_f1,
            'precision': best_precision,
            'recall': best_recall,
            'threshold': best_threshold,
        })
        print()

    return results


def experiment_original_state_propagation(data, W, num_layers, top1_final):
    """
    Original implementation: cos_sim(h_in, h_out) within same layer

    比較のため、元の実装（同一層内の h_in と h_out）も実験。
    """
    print("\n" + "=" * 60)
    print("Original State Propagation: cos_sim(h_in, h_out)")
    print("=" * 60)

    print("\n判定: Layer N の h_in と h_out の cos_sim が高い → Layer N で exit")
    print("予測対象: top1_layer_n == top1_final\n")

    results = []

    for layer in range(1, num_layers):
        print(f"--- Layer {layer} ---")

        h_out = torch.from_numpy(data[f'layer{layer}_h_out']).float().to(DEVICE)
        delta = torch.from_numpy(data[f'layer{layer}_delta']).float().to(DEVICE)
        h_in = h_out - delta

        # Compute cosine similarity
        h_in_norm = F.normalize(h_in, dim=-1)
        h_out_norm = F.normalize(h_out, dim=-1)
        cos_sim = (h_in_norm * h_out_norm).sum(dim=-1)

        # Labels
        top1_n = compute_top1_batched(h_out, W)
        labels = (top1_n == top1_final).float()
        saturation_rate = labels.mean().item()

        # Val split
        num_samples = h_out.shape[0]
        num_train = int(num_samples * 0.8)
        perm = torch.randperm(num_samples, device=DEVICE)
        val_idx = perm[num_train:]

        cos_sim_val = cos_sim[val_idx]
        labels_val = labels[val_idx].cpu().numpy()

        # Find best threshold
        best_f1 = 0
        best_threshold = 0

        for percentile in range(10, 100, 5):
            threshold = torch.quantile(cos_sim_val, percentile / 100.0).item()
            preds = (cos_sim_val > threshold).float().cpu().numpy()
            f1 = f1_score(labels_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Saturation rate: {saturation_rate * 100:.1f}%")
        print(f"cos_sim mean: {cos_sim.mean().item():.4f}")
        print(f"F1: {best_f1 * 100:.1f}%")

        results.append({
            'layer': layer,
            'saturation_rate': saturation_rate,
            'cos_sim_mean': cos_sim.mean().item(),
            'f1': best_f1,
        })
        print()

    return results


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
    calm_results = experiment_calm_state_propagation(data, W, num_layers, top1_final)
    original_results = experiment_original_state_propagation(data, W, num_layers, top1_final)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: CALM vs Original State Propagation")
    print("=" * 60)

    print(f"\n{'Layer':<8} {'Sat%':<8} {'CALM':<12} {'Original':<12} {'Diff':<10}")
    print("-" * 50)

    calm_f1s = []
    original_f1s = []

    for calm_r, orig_r in zip(calm_results, original_results):
        layer = calm_r['layer']
        sat = calm_r['saturation_rate'] * 100
        calm_f1 = calm_r['f1'] * 100
        orig_f1 = orig_r['f1'] * 100
        diff = calm_f1 - orig_f1

        calm_f1s.append(calm_f1)
        original_f1s.append(orig_f1)

        print(f"{layer:<8} {sat:<7.1f}% {calm_f1:<11.1f}% {orig_f1:<11.1f}% {diff:+.1f}%")

    print("-" * 50)

    avg_calm = np.mean(calm_f1s)
    avg_original = np.mean(original_f1s)
    avg_diff = avg_calm - avg_original

    print(f"{'Average':<8} {'':<8} {avg_calm:<11.1f}% {avg_original:<11.1f}% {avg_diff:+.1f}%")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    print(f"\nCALM State Propagation: cos_sim(h_n, h_n+1)")
    print(f"  - 連続2層間の類似度")
    print(f"  - 平均 F1 = {avg_calm:.1f}%")

    print(f"\nOriginal State Propagation: cos_sim(h_in, h_out)")
    print(f"  - 同一層内の入出力類似度")
    print(f"  - 平均 F1 = {avg_original:.1f}%")

    if avg_calm > avg_original:
        print(f"\n結論: CALM 方式が優位（+{avg_diff:.1f}%）")
        print(f"  → 連続層間の類似度が saturation 予測に有効")
    else:
        print(f"\n結論: Original 方式が優位（{avg_diff:.1f}%）")
        print(f"  → 同一層内の変化量が saturation 予測に有効")

    print(f"\n注意: どちらも MLP ベースの手法（h_out.delta: ~71%）より低い")


if __name__ == "__main__":
    main()
