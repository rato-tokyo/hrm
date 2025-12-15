"""
Analyze Residual-based Saturation Detection

残差接続の変化量（delta_norm）でtop-1トークンの変化を予測できるか検証。

仮説: delta_norm が小さい → 表現が変わらない → top-1 も変わらない → early exit可能

実験:
1. delta_norm と top-1変化の相関
2. delta_norm ベースの saturation 検出 F1
3. delta_norm vs MLP Router の比較
4. 各層での delta_norm 分布

Usage:
    python analyze_residual_saturation.py
    (requires layerwise_residuals.npz from save_layerwise_residuals.py)
"""

import numpy as np
import torch
from pathlib import Path
from scipy import stats


def load_data(path: str = "layerwise_residuals.npz"):
    """Load residual data."""
    print(f"Loading {path}...")
    data = np.load(path, allow_pickle=True)

    dim = int(data['dim'])
    vocab_size = int(data['vocab_size'])
    num_layers = int(data['num_layers'])

    print(f"Config: dim={dim}, vocab_size={vocab_size}, num_layers={num_layers}")

    return data, dim, vocab_size, num_layers


def compute_top1_batched(
    h: np.ndarray,
    W: np.ndarray,
    batch_size: int = 4096
) -> np.ndarray:
    """Compute top-1 predictions in batches to avoid OOM."""
    num_tokens = h.shape[0]
    top1_list = []

    for i in range(0, num_tokens, batch_size):
        h_batch = h[i:i + batch_size]
        logits_batch = h_batch @ W.T  # (batch, vocab_size)
        top1_batch = np.argmax(logits_batch, axis=-1)
        top1_list.append(top1_batch)

    return np.concatenate(top1_list)


def experiment1_correlation(data, num_layers, W):
    """実験1: delta_norm と top-1変化の相関"""
    print("\n" + "=" * 60)
    print("Experiment 1: Delta Norm vs Top-1 Change Correlation")
    print("=" * 60)

    results = []

    for i in range(num_layers - 1):
        layer_current = i + 1
        layer_next = i + 2

        print(f"\n--- Layer {layer_current} → Layer {layer_next} ---")

        # Get hidden states
        h_out_current = data[f'layer{layer_current}_h_out']
        h_out_next = data[f'layer{layer_next}_h_out']
        delta_next = data[f'layer{layer_next}_delta']

        # Compute delta norm (次の層での変化量)
        delta_norm = np.linalg.norm(delta_next, axis=-1)

        # Compute top-1 at each layer
        print("Computing top-1 predictions...")
        top1_current = compute_top1_batched(h_out_current, W)
        top1_next = compute_top1_batched(h_out_next, W)

        # Did top-1 change?
        top1_changed = (top1_current != top1_next).astype(np.float32)
        change_rate = top1_changed.mean()

        # Correlation: delta_norm vs top1_changed
        correlation, p_value = stats.pointbiserialr(delta_norm, top1_changed)

        print(f"Top-1 change rate: {change_rate * 100:.1f}%")
        print(f"Delta norm: mean={delta_norm.mean():.4f}, std={delta_norm.std():.4f}")
        print(f"Correlation (point-biserial): {correlation:.4f} (p={p_value:.2e})")

        # 分布比較: changed vs not changed
        delta_when_changed = delta_norm[top1_changed == 1]
        delta_when_stable = delta_norm[top1_changed == 0]

        print(f"Delta norm when top-1 CHANGED: mean={delta_when_changed.mean():.4f}")
        print(f"Delta norm when top-1 STABLE:  mean={delta_when_stable.mean():.4f}")

        results.append({
            'layer_transition': f"{layer_current}→{layer_next}",
            'change_rate': change_rate,
            'correlation': correlation,
            'delta_changed_mean': delta_when_changed.mean(),
            'delta_stable_mean': delta_when_stable.mean(),
        })

    return results


def experiment2_saturation_detection_f1(data, num_layers, W):
    """実験2: delta_norm ベースの saturation 検出 F1"""
    print("\n" + "=" * 60)
    print("Experiment 2: Saturation Detection F1 (Delta Norm)")
    print("=" * 60)

    results = []

    for i in range(num_layers - 1):
        layer_current = i + 1
        layer_next = i + 2

        print(f"\n--- Layer {layer_current} → Layer {layer_next} ---")

        # Get data
        h_out_current = data[f'layer{layer_current}_h_out']
        h_out_next = data[f'layer{layer_next}_h_out']
        delta_next = data[f'layer{layer_next}_delta']

        delta_norm = np.linalg.norm(delta_next, axis=-1)

        # Compute top-1
        top1_current = compute_top1_batched(h_out_current, W)
        top1_next = compute_top1_batched(h_out_next, W)

        # Ground truth: saturation = top-1 didn't change
        is_saturated = (top1_current == top1_next)

        # Try different thresholds
        thresholds = np.percentile(delta_norm, [10, 20, 30, 40, 50, 60, 70, 80, 90])

        best_f1 = 0
        best_threshold_percentile = 0

        for percentile, threshold in zip([10, 20, 30, 40, 50, 60, 70, 80, 90], thresholds):
            # Predict saturation: delta_norm < threshold
            predicted_saturated = delta_norm < threshold

            # Compute F1 for saturation detection
            tp = np.sum(predicted_saturated & is_saturated)
            fp = np.sum(predicted_saturated & ~is_saturated)
            fn = np.sum(~predicted_saturated & is_saturated)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold_percentile = percentile
                best_precision = precision
                best_recall = recall

        print(f"Best F1: {best_f1 * 100:.1f}% at {best_threshold_percentile}th percentile")
        print(f"  Precision: {best_precision * 100:.1f}%")
        print(f"  Recall: {best_recall * 100:.1f}%")

        results.append({
            'layer_transition': f"{layer_current}→{layer_next}",
            'best_f1': best_f1,
            'best_threshold_percentile': best_threshold_percentile,
        })

    return results


def experiment3_early_exit_simulation(data, num_layers, W):
    """実験3: delta_norm ベースの early exit シミュレーション"""
    print("\n" + "=" * 60)
    print("Experiment 3: Early Exit Simulation (Delta Norm)")
    print("=" * 60)

    print("\n仮定: delta_norm が小さいトークンは early exit")
    print("評価: exit したトークンの何%が実際に saturation していたか (Oracle %)")

    # Use final layer output as ground truth
    h_final = data[f'layer{num_layers}_h_out']
    top1_final = compute_top1_batched(h_final, W)

    for exit_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"\n--- Exit ratio: {exit_ratio * 100:.0f}% ---")

        for layer_idx in range(1, num_layers):
            layer = layer_idx + 1  # 1-indexed

            h_out = data[f'layer{layer}_h_out']
            delta = data[f'layer{layer}_delta']
            delta_norm = np.linalg.norm(delta, axis=-1)

            # Exit tokens with smallest delta_norm
            threshold = np.percentile(delta_norm, exit_ratio * 100)
            exit_mask = delta_norm < threshold

            # Top-1 at this layer
            top1_here = compute_top1_batched(h_out, W)

            # Of exited tokens, how many match final prediction?
            exited_match_final = (top1_here[exit_mask] == top1_final[exit_mask]).mean()

            print(f"  Layer {layer}: Exit {exit_mask.sum()} tokens, Oracle = {exited_match_final * 100:.1f}%")


def experiment4_delta_statistics(data, num_layers):
    """実験4: 各層での delta 統計"""
    print("\n" + "=" * 60)
    print("Experiment 4: Delta Statistics per Layer")
    print("=" * 60)

    print("\n各層でのdelta（残差）の統計:")
    print("-" * 50)

    for i in range(num_layers):
        layer = i + 1
        delta = data[f'layer{layer}_delta']
        delta_norm = np.linalg.norm(delta, axis=-1)

        h_in = data[f'layer{layer}_h_in']
        h_in_norm = np.linalg.norm(h_in, axis=-1)

        # Relative change
        relative_change = delta_norm / (h_in_norm + 1e-8)

        print(f"\nLayer {layer}:")
        print(f"  |delta|: mean={delta_norm.mean():.4f}, std={delta_norm.std():.4f}")
        print(f"  |h_in|:  mean={h_in_norm.mean():.4f}, std={h_in_norm.std():.4f}")
        print(f"  |delta|/|h_in| (relative): mean={relative_change.mean():.4f}, std={relative_change.std():.4f}")


def experiment5_cosine_similarity(data, num_layers, W):
    """実験5: cosine similarity による saturation 検出"""
    print("\n" + "=" * 60)
    print("Experiment 5: Cosine Similarity for Saturation Detection")
    print("=" * 60)

    print("\n仮説: h_in と h_out の cosine similarity が高い → 変化が少ない → saturation")

    results = []

    for i in range(num_layers - 1):
        layer_current = i + 1
        layer_next = i + 2

        print(f"\n--- Layer {layer_current} → Layer {layer_next} ---")

        # Get data
        h_in_next = data[f'layer{layer_next}_h_in']  # = h_out_current
        h_out_next = data[f'layer{layer_next}_h_out']

        # Cosine similarity between input and output of next layer
        h_in_norm = h_in_next / (np.linalg.norm(h_in_next, axis=-1, keepdims=True) + 1e-8)
        h_out_norm = h_out_next / (np.linalg.norm(h_out_next, axis=-1, keepdims=True) + 1e-8)
        cosine_sim = np.sum(h_in_norm * h_out_norm, axis=-1)

        # Compute top-1
        h_out_current = data[f'layer{layer_current}_h_out']
        top1_current = compute_top1_batched(h_out_current, W)
        top1_next = compute_top1_batched(h_out_next, W)

        is_saturated = (top1_current == top1_next)

        # Correlation
        correlation, p_value = stats.pointbiserialr(cosine_sim, is_saturated.astype(np.float32))

        print(f"Cosine similarity: mean={cosine_sim.mean():.4f}, std={cosine_sim.std():.4f}")
        print(f"Correlation with saturation: {correlation:.4f} (p={p_value:.2e})")

        # F1 score
        thresholds = np.percentile(cosine_sim, [50, 60, 70, 80, 90, 95])
        best_f1 = 0

        for percentile, threshold in zip([50, 60, 70, 80, 90, 95], thresholds):
            predicted_saturated = cosine_sim > threshold  # High similarity = saturated

            tp = np.sum(predicted_saturated & is_saturated)
            fp = np.sum(predicted_saturated & ~is_saturated)
            fn = np.sum(~predicted_saturated & is_saturated)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_percentile = percentile

        print(f"Best F1: {best_f1 * 100:.1f}% at {best_percentile}th percentile")

        results.append({
            'layer_transition': f"{layer_current}→{layer_next}",
            'correlation': correlation,
            'best_f1': best_f1,
        })

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
    W = data['output_head_W']

    print(f"\nOutput head W: {W.shape}")
    print(f"Number of tokens: {data['layer1_h_out'].shape[0]}")

    # Run experiments
    exp1_results = experiment1_correlation(data, num_layers, W)
    exp2_results = experiment2_saturation_detection_f1(data, num_layers, W)
    experiment3_early_exit_simulation(data, num_layers, W)
    experiment4_delta_statistics(data, num_layers)
    exp5_results = experiment5_cosine_similarity(data, num_layers, W)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n[Experiment 1] Delta Norm vs Top-1 Change Correlation:")
    for r in exp1_results:
        print(f"  {r['layer_transition']}: r={r['correlation']:.3f}, "
              f"delta_changed={r['delta_changed_mean']:.4f}, delta_stable={r['delta_stable_mean']:.4f}")

    print("\n[Experiment 2] Saturation Detection F1 (Delta Norm):")
    for r in exp2_results:
        print(f"  {r['layer_transition']}: F1={r['best_f1'] * 100:.1f}%")

    print("\n[Experiment 5] Saturation Detection F1 (Cosine Similarity):")
    for r in exp5_results:
        print(f"  {r['layer_transition']}: F1={r['best_f1'] * 100:.1f}%, r={r['correlation']:.3f}")

    print("\n結論:")
    avg_f1_delta = np.mean([r['best_f1'] for r in exp2_results])
    avg_f1_cosine = np.mean([r['best_f1'] for r in exp5_results])
    avg_correlation = np.mean([r['correlation'] for r in exp1_results])

    print(f"  - Delta norm 方式の平均 F1: {avg_f1_delta * 100:.1f}%")
    print(f"  - Cosine similarity 方式の平均 F1: {avg_f1_cosine * 100:.1f}%")
    print(f"  - Delta norm と top-1変化の平均相関: {avg_correlation:.3f}")

    if avg_correlation > 0.3:
        print("  → 残差の大きさと top-1 変化には有意な正の相関がある")
        print("  → delta_norm ベースの early exit は有望")
    elif avg_correlation > 0.1:
        print("  → 弱い正の相関はあるが、単独では不十分")
        print("  → 他の特徴量との組み合わせが必要かもしれない")
    else:
        print("  → 相関が弱い。残差ベースのアプローチは効果的でない可能性")


if __name__ == "__main__":
    main()
