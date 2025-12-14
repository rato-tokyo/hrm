"""
Mixture-of-Depths (MoD) Router Analysis

MoD方式のrouterをシミュレーションし、Easy/Hard分離性能を検証する。

MoD方式:
- Router: nn.Linear(dim, 1) → importance score
- 訓練: LM損失を通じて暗黙的に学習（出力 × importance で勾配を流す）
- 補助損失: なし

比較対象:
- 現在のLEGO: exit_classifierがlossを明示的に予測（Oracle 27%）
- MoD: routerが暗黙的にimportanceを学習

検証内容:
1. MoD-style router（linear層）を訓練
2. importance scoreでEasy/Hard分離
3. Oracle%を測定してmax_minus_mean（27%）と比較
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr


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

# ============================================================
# Load data
# ============================================================
print("\nLoading data...")
data = np.load("hidden_states_data.npz")
h = torch.from_numpy(data["hidden_states"]).to(device)  # (num_tokens, dim)
loss = torch.from_numpy(data["per_token_loss"]).to(device)  # (num_tokens,)
W = torch.from_numpy(data["output_head_W"]).to(device)  # (vocab_size, dim)

num_tokens, dim = h.shape
vocab_size = W.shape[0]

print(f"Hidden states: {h.shape}")
print(f"Per-token loss: {loss.shape}")
print(f"W matrix: {W.shape}")

# ============================================================
# Compute Oracle
# ============================================================
loss_cpu = loss.cpu().numpy()
oracle_threshold = np.median(loss_cpu)
oracle_easy = loss_cpu <= oracle_threshold
oracle_diff = loss_cpu[~oracle_easy].mean() - loss_cpu[oracle_easy].mean()
print(f"\nOracle diff: {oracle_diff:.2f}")
print(
    f"Oracle: Easy loss = {loss_cpu[oracle_easy].mean():.2f}, Hard loss = {loss_cpu[~oracle_easy].mean():.2f}"
)

# ============================================================
# Experiment 1: MoD-style Router Training
# ============================================================
print(f"\n{'=' * 60}")
print("Experiment 1: MoD-style Router Training")
print("=" * 60)
print("Router: nn.Linear(dim, 1) → importance score")
print("Training: MSE loss to predict (negative) loss")
print("(MoD uses implicit training, but we simulate with explicit loss prediction)")


class MoDRouter(nn.Module):
    """MoD-style router: single linear layer."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)  # (batch,) importance score


def evaluate_separation(
    importance: np.ndarray, loss_np: np.ndarray, oracle_diff_val: float
) -> dict:
    """Evaluate Easy/Hard separation quality."""
    threshold = np.median(importance)
    # Higher importance = easy (should have lower loss)
    easy_mask = importance >= threshold
    easy_loss = loss_np[easy_mask].mean()
    hard_loss = loss_np[~easy_mask].mean()
    diff = hard_loss - easy_loss
    oracle_pct = (diff / oracle_diff_val) * 100
    corr, _ = pearsonr(importance, loss_np)
    return {
        "easy_loss": easy_loss,
        "hard_loss": hard_loss,
        "diff": diff,
        "oracle_pct": oracle_pct,
        "corr": corr,
    }


# Train/test split
torch.manual_seed(42)
n_train = int(num_tokens * 0.8)
perm = torch.randperm(num_tokens)
train_idx = perm[:n_train]
test_idx = perm[n_train:]

h_train = h[train_idx]
h_test = h[test_idx]
loss_train = loss[train_idx]
loss_test = loss[test_idx]

print(f"\nTrain: {n_train}, Test: {num_tokens - n_train}")

# ============================================================
# Method 1: Linear router predicting negative loss (MoD simulation)
# ============================================================
print(f"\n{'=' * 60}")
print("Method 1: Linear Router (MoD-style)")
print("=" * 60)
print("Target: -loss (higher importance = lower loss = easy)")

router_linear = MoDRouter(dim).to(device)
optimizer = torch.optim.Adam(router_linear.parameters(), lr=1e-3)

# Target: negative loss (so higher output = easy)
target_train = -loss_train

batch_size = 4096
num_epochs = 50
best_val_loss = float("inf")
patience = 5
patience_counter = 0

start_time = time.time()
for epoch in range(num_epochs):
    router_linear.train()
    epoch_loss = 0.0
    num_batches = 0

    for i in range(0, n_train, batch_size):
        batch_h = h_train[i : i + batch_size]
        batch_target = target_train[i : i + batch_size]

        pred = router_linear(batch_h)
        batch_loss = F.mse_loss(pred, batch_target)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        num_batches += 1

    # Validation
    router_linear.eval()
    with torch.no_grad():
        val_pred = router_linear(h_test)
        val_loss = F.mse_loss(val_pred, -loss_test).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 10 == 0:
        print(
            f"  Epoch {epoch + 1}: train_loss={epoch_loss / num_batches:.4f}, val_loss={val_loss:.4f}"
        )

print(f"Training time: {time.time() - start_time:.1f}s")

# Evaluate
router_linear.eval()
with torch.no_grad():
    importance_linear = router_linear(h).cpu().numpy()

result_linear = evaluate_separation(importance_linear, loss_cpu, oracle_diff)
print("\nLinear Router Results:")
print(f"  Correlation with -loss: {-result_linear['corr']:.4f}")
print(f"  Easy Loss: {result_linear['easy_loss']:.2f}")
print(f"  Hard Loss: {result_linear['hard_loss']:.2f}")
print(f"  Diff: {result_linear['diff']:.2f}")
print(f"  Oracle %: {result_linear['oracle_pct']:.1f}%")

# ============================================================
# Method 2: MLP router (for comparison)
# ============================================================
print(f"\n{'=' * 60}")
print("Method 2: MLP Router (2-layer)")
print("=" * 60)


class MLPRouter(nn.Module):
    """2-layer MLP router."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(h))
        return self.fc2(x).squeeze(-1)


router_mlp = MLPRouter(dim, 128).to(device)
optimizer = torch.optim.Adam(router_mlp.parameters(), lr=1e-3)

best_val_loss = float("inf")
patience_counter = 0

start_time = time.time()
for epoch in range(num_epochs):
    router_mlp.train()
    epoch_loss = 0.0
    num_batches = 0

    for i in range(0, n_train, batch_size):
        batch_h = h_train[i : i + batch_size]
        batch_target = target_train[i : i + batch_size]

        pred = router_mlp(batch_h)
        batch_loss = F.mse_loss(pred, batch_target)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        num_batches += 1

    # Validation
    router_mlp.eval()
    with torch.no_grad():
        val_pred = router_mlp(h_test)
        val_loss = F.mse_loss(val_pred, -loss_test).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 10 == 0:
        print(
            f"  Epoch {epoch + 1}: train_loss={epoch_loss / num_batches:.4f}, val_loss={val_loss:.4f}"
        )

print(f"Training time: {time.time() - start_time:.1f}s")

# Evaluate
router_mlp.eval()
with torch.no_grad():
    importance_mlp = router_mlp(h).cpu().numpy()

result_mlp = evaluate_separation(importance_mlp, loss_cpu, oracle_diff)
print("\nMLP Router Results:")
print(f"  Correlation with -loss: {-result_mlp['corr']:.4f}")
print(f"  Easy Loss: {result_mlp['easy_loss']:.2f}")
print(f"  Hard Loss: {result_mlp['hard_loss']:.2f}")
print(f"  Diff: {result_mlp['diff']:.2f}")
print(f"  Oracle %: {result_mlp['oracle_pct']:.1f}%")

# ============================================================
# Method 3: Direct loss prediction (current LEGO approach)
# ============================================================
print(f"\n{'=' * 60}")
print("Method 3: Direct Loss Prediction (Current LEGO)")
print("=" * 60)
print("Target: loss directly (lower prediction = easy)")

router_lego = MLPRouter(dim, 128).to(device)
optimizer = torch.optim.Adam(router_lego.parameters(), lr=1e-3)

# Target: loss directly
target_train_lego = loss_train

best_val_loss = float("inf")
patience_counter = 0

start_time = time.time()
for epoch in range(num_epochs):
    router_lego.train()
    epoch_loss = 0.0
    num_batches = 0

    for i in range(0, n_train, batch_size):
        batch_h = h_train[i : i + batch_size]
        batch_target = target_train_lego[i : i + batch_size]

        pred = router_lego(batch_h)
        batch_loss = F.mse_loss(pred, batch_target)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        num_batches += 1

    # Validation
    router_lego.eval()
    with torch.no_grad():
        val_pred = router_lego(h_test)
        val_loss = F.mse_loss(val_pred, loss_test).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 10 == 0:
        print(
            f"  Epoch {epoch + 1}: train_loss={epoch_loss / num_batches:.4f}, val_loss={val_loss:.4f}"
        )

print(f"Training time: {time.time() - start_time:.1f}s")

# Evaluate (lower prediction = easy)
router_lego.eval()
with torch.no_grad():
    pred_loss = router_lego(h).cpu().numpy()

# For separation: lower predicted loss = easy
threshold = np.median(pred_loss)
easy_mask = pred_loss <= threshold
easy_loss_lego = loss_cpu[easy_mask].mean()
hard_loss_lego = loss_cpu[~easy_mask].mean()
diff_lego = hard_loss_lego - easy_loss_lego
oracle_pct_lego = (diff_lego / oracle_diff) * 100
corr_lego, _ = pearsonr(pred_loss, loss_cpu)

print("\nLEGO-style Router Results:")
print(f"  Correlation with loss: {corr_lego:.4f}")
print(f"  Easy Loss: {easy_loss_lego:.2f}")
print(f"  Hard Loss: {hard_loss_lego:.2f}")
print(f"  Diff: {diff_lego:.2f}")
print(f"  Oracle %: {oracle_pct_lego:.1f}%")

# ============================================================
# Comparison with max_minus_mean (no training needed)
# ============================================================
print(f"\n{'=' * 60}")
print("Comparison: max_minus_mean (no training)")
print("=" * 60)

# Compute max_minus_mean
chunk_size = 4000
num_chunks = (num_tokens + chunk_size - 1) // chunk_size
max_minus_mean = torch.zeros(num_tokens, device=device)

with torch.no_grad():
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_tokens)
        h_chunk = h[start:end]
        z_chunk = h_chunk @ W.T
        max_minus_mean[start:end] = z_chunk.max(dim=1).values - z_chunk.mean(dim=1)

max_minus_mean_cpu = max_minus_mean.cpu().numpy()
result_mmm = evaluate_separation(max_minus_mean_cpu, loss_cpu, oracle_diff)

print("max_minus_mean Results:")
print(f"  Correlation with -loss: {-result_mmm['corr']:.4f}")
print(f"  Easy Loss: {result_mmm['easy_loss']:.2f}")
print(f"  Hard Loss: {result_mmm['hard_loss']:.2f}")
print(f"  Diff: {result_mmm['diff']:.2f}")
print(f"  Oracle %: {result_mmm['oracle_pct']:.1f}%")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("Summary: Easy/Hard Separation Performance")
print("=" * 60)

print(
    f"\n{'Method':<30} {'Easy Loss':<12} {'Hard Loss':<12} {'Diff':<12} {'Oracle %':<12}"
)
print("-" * 78)
print(
    f"{'Linear Router (MoD)':<30} {result_linear['easy_loss']:<12.2f} {result_linear['hard_loss']:<12.2f} {result_linear['diff']:<12.2f} {result_linear['oracle_pct']:<12.1f}%"
)
print(
    f"{'MLP Router (MoD)':<30} {result_mlp['easy_loss']:<12.2f} {result_mlp['hard_loss']:<12.2f} {result_mlp['diff']:<12.2f} {result_mlp['oracle_pct']:<12.1f}%"
)
print(
    f"{'MLP (LEGO-style)':<30} {easy_loss_lego:<12.2f} {hard_loss_lego:<12.2f} {diff_lego:<12.2f} {oracle_pct_lego:<12.1f}%"
)
print(
    f"{'max_minus_mean (no training)':<30} {result_mmm['easy_loss']:<12.2f} {result_mmm['hard_loss']:<12.2f} {result_mmm['diff']:<12.2f} {result_mmm['oracle_pct']:<12.1f}%"
)
print(
    f"{'Oracle':<30} {loss_cpu[oracle_easy].mean():<12.2f} {loss_cpu[~oracle_easy].mean():<12.2f} {oracle_diff:<12.2f} {'100.0':<12}%"
)

print("""
Key Insights:
1. MoD Linear Router: Simplest approach (single linear layer)
2. MoD MLP Router: More expressive but more parameters
3. LEGO-style: Direct loss prediction
4. max_minus_mean: No training needed, uses logits statistics

Note: MoD's actual training is implicit (output × importance),
      but we simulate with explicit loss prediction for comparison.
""")
