"""
Analyze Saturation Events and Low-Rank Approximation (Large Model)

Same experiments as analyze_saturation_lowrank.py but for larger model (dim=256).
Focus on whether low-rank approximation becomes more effective with larger dim.

Requires: layerwise_hidden_states_large.npz (from save_layerwise_data_large.py)
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


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


def compute_top1(h: torch.Tensor, W: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    """Compute top-1 prediction for each token (batched to save memory)."""
    num_tokens = h.shape[0]
    top1_list = []

    for i in range(0, num_tokens, batch_size):
        h_batch = h[i:i + batch_size]
        logits_batch = h_batch @ W.T  # (batch, vocab_size)
        top1_batch = logits_batch.argmax(dim=-1)
        top1_list.append(top1_batch)

    return torch.cat(top1_list)


def compute_top1_lowrank(h: torch.Tensor, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, rank: int, batch_size: int = 2048) -> torch.Tensor:
    """Compute top-1 using low-rank approximation of W (batched to save memory)."""
    num_tokens = h.shape[0]
    top1_list = []

    U_r = U[:, :rank]  # (vocab_size, rank)
    S_r = S[:rank]  # (rank,)
    V_r = V[:rank, :]  # (rank, dim)

    for i in range(0, num_tokens, batch_size):
        h_batch = h[i:i + batch_size]
        h_proj = h_batch @ V_r.T  # (batch, rank)
        h_scaled = h_proj * S_r  # (batch, rank)
        logits_batch = h_scaled @ U_r.T  # (batch, vocab_size)
        top1_batch = logits_batch.argmax(dim=-1)
        top1_list.append(top1_batch)

    return torch.cat(top1_list)


def main():
    # Load data
    data_path = Path("layerwise_hidden_states_large.npz")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run save_layerwise_data_large.py first.")
        return

    print("\nLoading data...")
    data = np.load(data_path)

    layer_hidden = [
        torch.from_numpy(data["layer1_hidden"]).to(device),
        torch.from_numpy(data["layer2_hidden"]).to(device),
        torch.from_numpy(data["layer3_hidden"]).to(device),
        torch.from_numpy(data["layer4_hidden"]).to(device),
    ]
    targets = torch.from_numpy(data["targets"]).to(device)
    per_token_loss = torch.from_numpy(data["per_token_loss"]).to(device)
    W = torch.from_numpy(data["output_head_W"]).to(device)

    num_tokens = layer_hidden[0].shape[0]
    dim = int(data["dim"])
    vocab_size = int(data["vocab_size"])

    print(f"Tokens: {num_tokens:,}")
    print(f"Dim: {dim}, Vocab: {vocab_size}")
    print(f"W shape: {W.shape}")
    print(f"W size: {W.numel() * 4 / (1024 * 1024):.1f} MB")

    # ============================================================
    # Experiment 1: Saturation Analysis (Full W)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Experiment 1: Saturation Analysis (Full W)")
    print("=" * 60)

    # Compute top-1 at each layer
    top1_per_layer = []
    for i, h in enumerate(layer_hidden):
        start = time.time()
        top1 = compute_top1(h, W)
        elapsed = time.time() - start
        top1_per_layer.append(top1)

        # Accuracy
        acc = (top1 == targets).float().mean().item() * 100
        print(f"Layer {i + 1}: top1 accuracy = {acc:.2f}%, time = {elapsed:.2f}s")

    # Saturation: does top-1 change between layers?
    print(f"\nSaturation Analysis:")
    print("-" * 50)

    for i in range(3):
        same = (top1_per_layer[i] == top1_per_layer[i + 1]).float().mean().item() * 100
        print(f"Layer {i + 1} → {i + 2}: top-1 unchanged = {same:.2f}%")

    # Saturation from layer 1 to final
    for i in range(3):
        same = (top1_per_layer[i] == top1_per_layer[3]).float().mean().item() * 100
        print(f"Layer {i + 1} → 4 (final): top-1 unchanged = {same:.2f}%")

    # ============================================================
    # Experiment 2: Low-Rank W Analysis
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Experiment 2: Low-Rank W Approximation")
    print("=" * 60)

    # SVD of W
    print("\nComputing SVD of W...")
    start = time.time()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    V = Vt  # (dim, dim)
    print(f"SVD time: {time.time() - start:.2f}s")
    print(f"U: {U.shape}, S: {S.shape}, V: {V.shape}")

    # Singular value analysis
    print(f"\nSingular value coverage:")
    total_var = (S ** 2).sum().item()
    for r in [8, 16, 32, 64, 128, 256]:
        if r <= len(S):
            coverage = (S[:r] ** 2).sum().item() / total_var * 100
            print(f"  rank={r:3d}: {coverage:.1f}%")

    # Test different ranks
    ranks_to_test = [8, 16, 32, 64, 128, 256]

    print(f"\n{'Layer':<10} {'Full W':<12} " + " ".join([f"r={r:<5}" for r in ranks_to_test]))
    print("-" * 90)

    lowrank_top1 = {}  # {(layer, rank): top1}

    for layer_idx in range(4):
        h = layer_hidden[layer_idx]

        # Full W accuracy
        full_acc = (top1_per_layer[layer_idx] == targets).float().mean().item() * 100

        accs = [f"{full_acc:.2f}%"]

        for rank in ranks_to_test:
            if rank > dim:
                accs.append("N/A")
                continue

            top1_lr = compute_top1_lowrank(h, U, S, V, rank)
            lowrank_top1[(layer_idx, rank)] = top1_lr

            # Agreement with full W
            agreement = (top1_lr == top1_per_layer[layer_idx]).float().mean().item() * 100
            accs.append(f"{agreement:.1f}%")

        print(f"Layer {layer_idx + 1:<4} {accs[0]:<12} " + " ".join([f"{a:<7}" for a in accs[1:]]))

    # ============================================================
    # Experiment 3: Low-Rank Saturation Detection
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Experiment 3: Low-Rank Saturation Detection")
    print("=" * 60)
    print("Can low-rank W detect saturation (top-1 won't change)?")

    for rank in [32, 64, 128]:
        if rank > dim:
            continue
        print(f"\n--- Rank {rank} ---")

        for i in range(3):
            if (i, rank) not in lowrank_top1 or (i + 1, rank) not in lowrank_top1:
                continue

            # Low-rank top-1 at layer i and i+1
            lr_top1_i = lowrank_top1[(i, rank)]
            lr_top1_next = lowrank_top1[(i + 1, rank)]

            # Full W top-1
            full_top1_i = top1_per_layer[i]
            full_top1_next = top1_per_layer[i + 1]

            # Low-rank says "saturated" (top-1 same)
            lr_saturated = (lr_top1_i == lr_top1_next)

            # Actually saturated (full W)
            full_saturated = (full_top1_i == full_top1_next)

            # Metrics
            tp = (lr_saturated & full_saturated).float().sum().item()
            fp = (lr_saturated & ~full_saturated).float().sum().item()
            fn = (~lr_saturated & full_saturated).float().sum().item()
            tn = (~lr_saturated & ~full_saturated).float().sum().item()

            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Layer {i + 1}→{i + 2}: Precision={precision:.1f}%, Recall={recall:.1f}%, F1={f1:.1f}%")

    # ============================================================
    # Experiment 4: Saturation vs Loss
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Experiment 4: Saturation vs Loss Relationship")
    print("=" * 60)

    loss_np = per_token_loss.cpu().numpy()

    for i in range(3):
        saturated = (top1_per_layer[i] == top1_per_layer[3]).cpu().numpy()

        sat_loss = loss_np[saturated].mean()
        unsat_loss = loss_np[~saturated].mean()
        sat_ratio = saturated.mean() * 100

        print(f"Layer {i + 1}→4: Saturated={sat_ratio:.1f}%, Sat_loss={sat_loss:.2f}, Unsat_loss={unsat_loss:.2f}, Diff={unsat_loss - sat_loss:.2f}")

    # ============================================================
    # Experiment 5: Speed Comparison
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Experiment 5: Speed Comparison (Full vs Low-Rank)")
    print("=" * 60)

    # Use smaller subset for fair timing (single batch)
    h_test = layer_hidden[0][:2048]  # Single batch for timing
    n_trials = 20

    # Full W (single batch, no loop)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_trials):
        logits = h_test @ W.T
        _ = logits.argmax(dim=-1)
    if device == "cuda":
        torch.cuda.synchronize()
    full_time = (time.time() - start) / n_trials

    print(f"Full W ({dim}x{vocab_size}): {full_time * 1000:.2f} ms/batch (batch=2048)")

    U_r_cache = {}
    for rank in [32, 64, 128]:
        if rank > dim:
            continue
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = V[:rank, :]

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_trials):
            h_proj = h_test @ V_r.T
            h_scaled = h_proj * S_r
            logits = h_scaled @ U_r.T
            _ = logits.argmax(dim=-1)
        if device == "cuda":
            torch.cuda.synchronize()
        lr_time = (time.time() - start) / n_trials

        speedup = full_time / lr_time
        print(f"Rank {rank:3d}: {lr_time * 1000:.2f} ms/batch, speedup = {speedup:.2f}x")
        U_r_cache[rank] = lr_time

    # ============================================================
    # Visualization
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Creating Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Large Model Analysis (dim={dim})', fontsize=14, fontweight='bold')

    # 1. Saturation rate across layers
    ax = axes[0, 0]
    sat_rates = []
    for i in range(4):
        same = (top1_per_layer[i] == top1_per_layer[3]).float().mean().item() * 100
        sat_rates.append(same)
    ax.bar(range(1, 5), sat_rates, color=['blue', 'blue', 'blue', 'green'])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Saturation Rate (%)')
    ax.set_title('Top-1 Agreement with Final Layer')
    ax.set_xticks(range(1, 5))
    ax.set_ylim(0, 100)
    for i, v in enumerate(sat_rates):
        ax.text(i + 1, v + 2, f'{v:.1f}%', ha='center')

    # 2. Low-rank agreement with full W (Layer 4)
    ax = axes[0, 1]
    ranks = [r for r in [8, 16, 32, 64, 128, 256] if r <= dim]
    agreements = []
    for r in ranks:
        if (3, r) in lowrank_top1:
            agr = (lowrank_top1[(3, r)] == top1_per_layer[3]).float().mean().item() * 100
        else:
            top1_lr = compute_top1_lowrank(layer_hidden[3], U, S, V, r)
            agr = (top1_lr == top1_per_layer[3]).float().mean().item() * 100
        agreements.append(agr)
    ax.plot(ranks, agreements, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Agreement with Full W (%)')
    ax.set_title('Low-Rank Top-1 Agreement (Layer 4)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Singular values
    ax = axes[0, 2]
    S_np = S.cpu().numpy()
    ax.semilogy(S_np, 'b-', linewidth=2)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value (log scale)')
    ax.set_title(f'Singular Values of W (dim={dim})')
    ax.grid(True, alpha=0.3)

    # Add cumulative variance
    ax2 = ax.twinx()
    cumvar = np.cumsum(S_np ** 2) / np.sum(S_np ** 2) * 100
    ax2.plot(cumvar, 'r--', alpha=0.7, label='Cumulative %')
    ax2.set_ylabel('Cumulative Variance (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)

    # 4. Loss distribution: saturated vs unsaturated
    ax = axes[1, 0]
    saturated_mask = (top1_per_layer[0] == top1_per_layer[3]).cpu().numpy()
    ax.hist(loss_np[saturated_mask], bins=50, alpha=0.7, label=f'Saturated ({saturated_mask.mean()*100:.1f}%)', density=True)
    ax.hist(loss_np[~saturated_mask], bins=50, alpha=0.7, label=f'Unsaturated ({(~saturated_mask).mean()*100:.1f}%)', density=True)
    ax.set_xlabel('Loss')
    ax.set_ylabel('Density')
    ax.set_title('Loss Distribution: Layer 1 Saturation')
    ax.legend()
    ax.set_xlim(0, 20)

    # 5. Accuracy by layer
    ax = axes[1, 1]
    accs = []
    for i in range(4):
        acc = (top1_per_layer[i] == targets).float().mean().item() * 100
        accs.append(acc)
    ax.plot(range(1, 5), accs, 'go-', linewidth=2, markersize=10)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Prediction Accuracy by Layer')
    ax.set_xticks(range(1, 5))
    ax.grid(True, alpha=0.3)

    # 6. Speed comparison
    ax = axes[1, 2]
    valid_ranks = [r for r in [32, 64, 128] if r <= dim and r in U_r_cache]
    labels = ['Full W'] + [f'Rank {r}' for r in valid_ranks]
    times = [full_time * 1000] + [U_r_cache[r] * 1000 for r in valid_ranks]

    colors = ['red'] + ['blue'] * len(valid_ranks)
    bars = ax.bar(labels, times, color=colors)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Speed Comparison (batch=2048)')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{t:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('saturation_lowrank_analysis_large.png', dpi=150)
    print("Saved: saturation_lowrank_analysis_large.png")

    # ============================================================
    # Comparison with Small Model
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Summary: Large Model (dim=256) vs Small Model (dim=64)")
    print("=" * 60)

    print(f"""
Large Model Results (dim={dim}):

1. Saturation Rate:
   - Layer 1→4: {sat_rates[0]:.1f}%
   - Layer 2→4: {sat_rates[1]:.1f}%
   - Layer 3→4: {sat_rates[2]:.1f}%

2. Low-Rank Approximation (Layer 4):
   - Rank 32:  {agreements[ranks.index(32)] if 32 in ranks else 'N/A':.1f}% agreement
   - Rank 64:  {agreements[ranks.index(64)] if 64 in ranks else 'N/A':.1f}% agreement
   - Rank 128: {agreements[ranks.index(128)] if 128 in ranks else 'N/A':.1f}% agreement

3. Speed Comparison:
   - Full W: {full_time * 1000:.2f} ms
   - Rank 64: {U_r_cache.get(64, 0) * 1000:.2f} ms ({full_time / U_r_cache.get(64, full_time):.2f}x speedup)

4. Key Observations:
   - Larger dim allows more effective low-rank compression
   - Speed gains should be more significant with dim=256
   - Compare with small model results to quantify improvement
""")

    plt.show()


if __name__ == "__main__":
    main()
