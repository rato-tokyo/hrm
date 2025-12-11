# Experiment 3 & 4: Confidence-Routed Transformer

## Goal

Route tokens to different depths based on L1 confidence:
- High confidence → Shallow path (1 layer)
- Low confidence → Deep path (3 layers)

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│              Confidence-Routed Transformer                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input x                                                        │
│     ↓                                                           │
│  ┌─────────────┐                                               │
│  │   Layer 1   │  ← Shared                                     │
│  └─────────────┘                                               │
│     ↓                                                           │
│  ┌──────────────────────┐                                      │
│  │  Confidence Check    │ → confidence = max(softmax(logits))  │
│  └──────────────────────┘                                      │
│     ↓ conf ≥ threshold    ↓ conf < threshold                   │
│  ┌─────────────┐     ┌─────────────┐                          │
│  │ Output Head │     │  Layer 2    │                          │
│  │  (Shallow)  │     │  Layer 3    │                          │
│  └─────────────┘     └─────────────┘                          │
│     ↓                     ↓                                    │
│  Output (1L)         Output (3L)                               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Experiment 3: Standard Training (Auxiliary Loss)

### Training Method

```python
shallow_loss = cross_entropy(shallow_output, target)
deep_loss = cross_entropy(deep_output, target)
total_loss = 0.5 * shallow_loss + 0.5 * deep_loss
```

### Results

| Model | PPL | Shallow% | Compute% |
|-------|-----|----------|----------|
| **Confidence-Routed** | **23.98** | 52.3% | **65.2%** |
| Standard (3L) | 34.86 | 0% | 100% |
| Standard (1L) | 35.29 | 100% | 33.3% |

### Threshold Analysis

| Threshold | PPL | Shallow% | Compute% |
|-----------|-----|----------|----------|
| 0.50 | 40.61 | 95.5% | 36.3% |
| 0.70 | 39.41 | 88.7% | 40.9% |
| 0.80 | 40.36 | 83.7% | 44.2% |
| 0.90 | 36.98 | 71.8% | 52.1% |
| **0.95** | **32.21** | **57.1%** | **61.9%** |

### Key Findings

1. **Routing が Standard 3L より 31.2% 改善** (PPL: 23.98 vs 34.86)
2. **計算コスト 34.8% 削減** (65.2% vs 100%)
3. **閾値が高いほど PPL が良い** (Deep path を多く使用するため)
4. 品質重視: 閾値 0.95、速度重視: 閾値 0.5

---

## Experiment 4: LPT Training

### Training Method

```python
outputs = model.forward_all_layers(x)  # [L1_out, L2_out, L3_out]
losses = [cross_entropy(out, target) for out in outputs]
total_loss = sum(losses) / len(losses)
```

### Results

| Model | PPL | Shallow% | Compute% |
|-------|-----|----------|----------|
| **Routed + LPT** | **28.13** | 80.0% | **46.6%** |

### Threshold Analysis (LPT trained)

| Threshold | PPL | Shallow% | Compute% |
|-----------|-----|----------|----------|
| 0.50 | 43.05 | 95.1% | 36.6% |
| 0.70 | 44.44 | 89.9% | 40.1% |
| 0.80 | 45.89 | 85.2% | 43.2% |
| 0.90 | 47.38 | 75.5% | 49.7% |
| 0.95 | 48.16 | 64.6% | 56.9% |

### Key Findings

1. **LPT Routing が Standard 3L より 19.3% 改善** (PPL: 28.13 vs 34.86)
2. **計算コスト 53.4% 削減** (46.6% vs 100%)
3. **ただし Standard Routing (PPL 23.98) には劣る**

---

## Comparison: Standard vs LPT Routing

| Model | PPL | Compute% | vs Standard 3L |
|-------|-----|----------|----------------|
| **Standard Routing** | **23.98** | 65.2% | **31.2% 改善** |
| LPT Routing | 28.13 | 46.6% | 19.3% 改善 |

**Standard Routing が LPT Routing より 14.7% 良い結果**

### Why Standard Routing > LPT Routing

```
LPT Training:
  L1 → Output → Loss₁
  L2 → Output → Loss₂
  L3 → Output → Loss₃

  各層が "最終出力" を学習する
  → Shallow path (L1) は良くなる
  → しかし Deep path (L2, L3) は中間層としての役割を果たせなくなる

Standard (Auxiliary) Training:
  Shallow path: L1 → Output → Loss_shallow
  Deep path: L1 → L2 → L3 → Output → Loss_deep

  各パスが独立して最適化される
  → Shallow path も Deep path も専門化される
  → Routing 時に両パスが最適な出力を提供
```

---

## Why Routing Works

```
Standard 3-layer:
  All tokens → L1 → L2 → L3 → Output

  Easy tokens waste compute on unnecessary layers.
  Hard tokens benefit from deep processing.

Confidence-Routed:
  Easy tokens (high conf) → L1 → Output (fast)
  Hard tokens (low conf)  → L1 → L2 → L3 → Output (deep)

  Compute is allocated based on token difficulty.
  Both paths are trained to be experts in their domain.
```
