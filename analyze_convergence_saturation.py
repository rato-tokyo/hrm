"""
Analyze Convergence and Saturation Relationship

仮説: x（hidden states）が収束 → saturation（top-1が固定）

実験1: 層ごとの delta_norm と saturation 率の関係を可視化

Usage:
    python analyze_convergence_saturation.py
"""

import numpy as np
import torch
from pathlib import Path


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

    num_tokens = data['layer1_h_out'].shape[0]
    print(f"\nNumber of tokens: {num_tokens}")

    # Compute top-1 for final layer (ground truth)
    print("\nComputing top-1 for each layer...")
    h_final = torch.from_numpy(data[f'layer{num_layers}_h_out']).to(DEVICE)
    top1_final = compute_top1_batched(h_final, W)

    # Collect per-layer statistics
    layer_stats = []

    for layer in range(1, num_layers + 1):
        # Load data for this layer
        h_out = torch.from_numpy(data[f'layer{layer}_h_out']).to(DEVICE)
        delta = torch.from_numpy(data[f'layer{layer}_delta']).to(DEVICE)

        # Compute delta_norm
        delta_norm = torch.norm(delta, dim=-1)
        mean_delta_norm = delta_norm.mean().item()
        std_delta_norm = delta_norm.std().item()

        # Compute top-1 at this layer
        top1_here = compute_top1_batched(h_out, W)

        # Saturation rate: top-1 matches final layer
        saturation_rate = (top1_here == top1_final).float().mean().item()

        layer_stats.append({
            'layer': layer,
            'mean_delta_norm': mean_delta_norm,
            'std_delta_norm': std_delta_norm,
            'saturation_rate': saturation_rate,
        })

        print(f"Layer {layer}: delta_norm={mean_delta_norm:.4f} (±{std_delta_norm:.4f}), saturation={saturation_rate * 100:.1f}%")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Layer-wise Convergence and Saturation")
    print("=" * 60)

    print("\n{:<8} {:>15} {:>15} {:>15}".format(
        "Layer", "Delta Norm", "Std", "Saturation %"
    ))
    print("-" * 55)

    for stats in layer_stats:
        print(f"{stats['layer']:<8} {stats['mean_delta_norm']:>15.4f} {stats['std_delta_norm']:>15.4f} {stats['saturation_rate'] * 100:>14.1f}%")

    # Analyze correlation between delta_norm and saturation
    print("\n" + "=" * 60)
    print("ANALYSIS: Convergence vs Saturation")
    print("=" * 60)

    delta_norms = [s['mean_delta_norm'] for s in layer_stats]
    saturation_rates = [s['saturation_rate'] for s in layer_stats]

    # Correlation (excluding last layer which may be special)
    from scipy import stats as scipy_stats

    # Full correlation
    corr_full, p_full = scipy_stats.pearsonr(delta_norms, saturation_rates)
    print(f"\nCorrelation (all layers): r={corr_full:.3f}, p={p_full:.4f}")

    # Excluding last layer
    if num_layers > 2:
        corr_excl, p_excl = scipy_stats.pearsonr(delta_norms[:-1], saturation_rates[:-1])
        print(f"Correlation (excl. last layer): r={corr_excl:.3f}, p={p_excl:.4f}")

    # Token-level analysis
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL ANALYSIS")
    print("=" * 60)

    print("\nFor each layer, correlation between token's delta_norm and whether it's saturated:")

    for layer in range(1, num_layers + 1):
        h_out = torch.from_numpy(data[f'layer{layer}_h_out']).to(DEVICE)
        delta = torch.from_numpy(data[f'layer{layer}_delta']).to(DEVICE)

        delta_norm = torch.norm(delta, dim=-1).cpu().numpy()
        top1_here = compute_top1_batched(h_out, W)
        is_saturated = (top1_here == top1_final).cpu().numpy().astype(float)

        # Point-biserial correlation
        corr, p_value = scipy_stats.pointbiserialr(delta_norm, is_saturated)
        print(f"  Layer {layer}: r={corr:.3f} (p={p_value:.2e})")

    # Cumulative delta analysis
    print("\n" + "=" * 60)
    print("CUMULATIVE DELTA ANALYSIS")
    print("=" * 60)

    print("\nDoes cumulative delta predict saturation better?")

    cumulative_delta = np.zeros(num_tokens)

    for layer in range(1, num_layers + 1):
        delta = torch.from_numpy(data[f'layer{layer}_delta']).to(DEVICE)
        delta_norm = torch.norm(delta, dim=-1).cpu().numpy()
        cumulative_delta += delta_norm

        h_out = torch.from_numpy(data[f'layer{layer}_h_out']).to(DEVICE)
        top1_here = compute_top1_batched(h_out, W)
        is_saturated = (top1_here == top1_final).cpu().numpy().astype(float)

        # Correlation with cumulative delta
        corr, p_value = scipy_stats.pointbiserialr(cumulative_delta, is_saturated)
        print(f"  Layer {layer} (cumulative): r={corr:.3f} (p={p_value:.2e})")

    # Delta decay analysis
    print("\n" + "=" * 60)
    print("DELTA DECAY ANALYSIS")
    print("=" * 60)

    print("\nDo tokens with decreasing delta tend to saturate?")

    # Compare delta at layer 1 vs layer N-1 (excluding last layer)
    delta_layer1 = torch.norm(
        torch.from_numpy(data['layer1_delta']).to(DEVICE), dim=-1
    ).cpu().numpy()

    last_normal_layer = num_layers - 1  # Exclude last layer (often special)
    delta_layer_last = torch.norm(
        torch.from_numpy(data[f'layer{last_normal_layer}_delta']).to(DEVICE), dim=-1
    ).cpu().numpy()

    # Decay ratio
    decay_ratio = delta_layer_last / (delta_layer1 + 1e-8)

    # Is saturated at last normal layer?
    h_out = torch.from_numpy(data[f'layer{last_normal_layer}_h_out']).to(DEVICE)
    top1_here = compute_top1_batched(h_out, W)
    is_saturated = (top1_here == top1_final).cpu().numpy().astype(float)

    corr, p_value = scipy_stats.pointbiserialr(decay_ratio, is_saturated)
    print(f"\nDecay ratio (Layer {last_normal_layer} / Layer 1) vs Saturation:")
    print(f"  Correlation: r={corr:.3f} (p={p_value:.2e})")

    # Statistics for saturated vs non-saturated tokens
    saturated_mask = is_saturated == 1
    decay_saturated = decay_ratio[saturated_mask].mean()
    decay_not_saturated = decay_ratio[~saturated_mask].mean()

    print(f"\n  Mean decay ratio for SATURATED tokens: {decay_saturated:.4f}")
    print(f"  Mean decay ratio for NON-SATURATED tokens: {decay_not_saturated:.4f}")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if corr_full < -0.5:
        print("\n強い負の相関: delta_norm が小さい → saturation 率が高い")
        print("→ 仮説「x の収束 → saturation」は支持される")
    elif corr_full < -0.2:
        print("\n弱〜中程度の負の相関")
        print("→ 仮説は部分的に支持されるが、他の要因も影響している可能性")
    elif corr_full > 0.2:
        print("\n正の相関（予想と逆）")
        print("→ 仮説は支持されない。最終層の特殊性が影響している可能性")
    else:
        print("\n相関が弱い")
        print("→ delta_norm と saturation の関係は単純ではない")


if __name__ == "__main__":
    main()
