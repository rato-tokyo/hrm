"""
Analyze Saturation Events and Low-Rank Approximation

Experiments:
1. Saturation Analysis: Does top-1 prediction change between layers?
2. Low-Rank W: Can we use low-rank W to predict top-1 accurately?
3. Combined: Low-rank saturation detection

Requires: layerwise_hidden_states.npz (from save_layerwise_data.py)
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


def compute_top1(h: torch.Tensor, W: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """Compute top-1 prediction for each token (batched to save memory)."""
    num_tokens = h.shape[0]
    top1_list = []

    for i in range(0, num_tokens, batch_size):
        h_batch = h[i:i + batch_size]
        logits_batch = h_batch @ W.T  # (batch, vocab_size)
        top1_batch = logits_batch.argmax(dim=-1)
        top1_list.append(top1_batch)

    return torch.cat(top1_list)


def compute_top1_lowrank(h: torch.Tensor, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, rank: int, batch_size: int = 4096) -> torch.Tensor:
    """Compute top-1 using low-rank approximation of W (batched to save memory)."""
    num_tokens = h.shape[0]
    top1_list = []

    U_r = U[:, :rank]  # (vocab_size, rank)
    S_r = S[:rank]  # (rank,)
    V_r = V[:rank, :]  # (rank, dim)

    for i in range(0, num_tokens, batch_size):
        h_batch = h[i:i + batch_size]
        # W_lowrank = U_r @ diag(S_r) @ V_r
        # logits = h @ W_lowrank.T = h @ V_r.T @ diag(S_r) @ U_r.T
        h_proj = h_batch @ V_r.T  # (batch, rank)
        h_scaled = h_proj * S_r  # (batch, rank)
        logits_batch = h_scaled @ U_r.T  # (batch, vocab_size)
        top1_batch = logits_batch.argmax(dim=-1)
        top1_list.append(top1_batch)

    return torch.cat(top1_list)


def main():
    # Load data
    data_path = Path("layerwise_hidden_states.npz")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run save_layerwise_data.py first.")
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
    print(f"\n{'Saturation Analysis':}")
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
    V = Vt  # (dim, dim) - already transposed correctly
    print(f"SVD time: {time.time() - start:.2f}s")
    print(f"U: {U.shape}, S: {S.shape}, V: {V.shape}")

    # Singular value analysis
    print(f"\nSingular value coverage:")
    total_var = (S ** 2).sum().item()
    for r in [4, 8, 16, 32, 64]:
        if r <= len(S):
            coverage = (S[:r] ** 2).sum().item() / total_var * 100
            print(f"  rank={r:2d}: {coverage:.1f}%")

    # Test different ranks
    ranks_to_test = [4, 8, 16, 32, 64]

    print(f"\n{'Layer':<10} {'Full W':<12} " + " ".join([f"rank={r:<4}" for r in ranks_to_test]))
    print("-" * 70)

    lowrank_top1 = {}  # {(layer, rank): top1}

    for layer_idx in range(4):
        h = layer_hidden[layer_idx]

        # Full W accuracy
        full_acc = (top1_per_layer[layer_idx] == targets).float().mean().item() * 100

        accs = [f"{full_acc:.2f}%"]

        for rank in ranks_to_test:
            top1_lr = compute_top1_lowrank(h, U, S, V, rank)
            lowrank_top1[(layer_idx, rank)] = top1_lr

            # Agreement with full W
            agreement = (top1_lr == top1_per_layer[layer_idx]).float().mean().item() * 100
            accs.append(f"{agreement:.1f}%")

        print(f"Layer {layer_idx + 1:<4} {accs[0]:<12} " + " ".join([f"{a:<9}" for a in accs[1:]]))

    # ============================================================
    # Experiment 3: Low-Rank Saturation Detection
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Experiment 3: Low-Rank Saturation Detection")
    print("=" * 60)
    print("Can low-rank W detect saturation (top-1 won't change)?")

    for rank in [8, 16, 32]:
        print(f"\n--- Rank {rank} ---")

        for i in range(3):
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
            # True positive: LR says saturated AND actually saturated
            tp = (lr_saturated & full_saturated).float().sum().item()
            # False positive: LR says saturated BUT actually changed
            fp = (lr_saturated & ~full_saturated).float().sum().item()
            # False negative: LR says changed BUT actually saturated
            fn = (~lr_saturated & full_saturated).float().sum().item()
            # True negative: LR says changed AND actually changed
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
    print("Are saturated tokens (top-1 unchanged) easier (lower loss)?")

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
    h_test = layer_hidden[0][:4096]  # Single batch for timing
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

    print(f"Full W ({dim}x{vocab_size}): {full_time * 1000:.2f} ms/batch (batch=4096)")

    U_r_cache = {}
    for rank in [8, 16, 32]:
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
        print(f"Rank {rank:2d}: {lr_time * 1000:.2f} ms/batch, speedup = {speedup:.2f}x")
        U_r_cache[rank] = lr_time

    # ============================================================
    # Visualization
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Creating Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Saturation rate across layers
    ax = axes[0, 0]
    sat_rates = []
    for i in range(4):
        same = (top1_per_layer[i] == top1_per_layer[3]).float().mean().item() * 100
        sat_rates.append(same)
    ax.bar(range(1, 5), sat_rates)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Saturation Rate (%)')
    ax.set_title('Top-1 Agreement with Final Layer')
    ax.set_xticks(range(1, 5))
    ax.set_ylim(0, 100)
    for i, v in enumerate(sat_rates):
        ax.text(i + 1, v + 2, f'{v:.1f}%', ha='center')

    # 2. Low-rank agreement with full W
    ax = axes[0, 1]
    ranks = [4, 8, 16, 32, 64]
    agreements = []
    for r in ranks:
        top1_lr = compute_top1_lowrank(layer_hidden[3], U, S, V, r)
        agr = (top1_lr == top1_per_layer[3]).float().mean().item() * 100
        agreements.append(agr)
    ax.plot(ranks, agreements, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Agreement with Full W (%)')
    ax.set_title('Low-Rank Top-1 Agreement (Layer 4)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # 3. Singular values
    ax = axes[0, 2]
    S_np = S.cpu().numpy()
    ax.semilogy(S_np, 'b-', linewidth=2)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value (log scale)')
    ax.set_title('Singular Values of W')
    ax.grid(True, alpha=0.3)

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

    # 6. Speed comparison (use cached times from Experiment 5)
    ax = axes[1, 2]
    labels = ['Full W'] + [f'Rank {r}' for r in [8, 16, 32]]
    times = [full_time * 1000] + [U_r_cache[r] * 1000 for r in [8, 16, 32]]

    bars = ax.bar(labels, times, color=['red', 'blue', 'blue', 'blue'])
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Speed Comparison (batch=4096)')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{t:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig('saturation_lowrank_analysis.png', dpi=150)
    print("Saved: saturation_lowrank_analysis.png")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)

    print(f"""
Key Findings:

1. Saturation Rate:
   - Layer 1→4 saturation: {sat_rates[0]:.1f}%
   - Layer 2→4 saturation: {sat_rates[1]:.1f}%
   - Layer 3→4 saturation: {sat_rates[2]:.1f}%
   - Earlier layers already determine many final predictions

2. Low-Rank Approximation:
   - Rank 32 achieves {agreements[ranks.index(32)]:.1f}% agreement with full W
   - Significant speedup possible with minimal accuracy loss

3. Saturation vs Loss:
   - Saturated tokens have LOWER loss (easier tokens)
   - This validates early exit for easy tokens

4. Implications for LEGO:
   - Can predict "will top-1 change?" instead of "what is the loss?"
   - Low-rank W can accelerate exit decision
   - Early saturation = early exit opportunity
""")

    plt.show()


if __name__ == "__main__":
    main()
