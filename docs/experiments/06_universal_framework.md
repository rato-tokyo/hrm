# Universal Training Framework

## Overview

すべての学習方法を統一的に表現できるフレームワーク。
各層への重み付けと推論時のRouting設定を分離して管理。

## Core Concept

```
Loss = Σ weights[i] * L_i_loss

where:
  weights[i] = layer i の重み (0以上の任意の値)
  L_i_loss = layer i の出力に対する損失
```

## Configuration

```python
@dataclass
class UniversalConfig:
    layer_weights: Dict[int, float]  # {layer_idx: weight} (1-indexed)
    exit_layer: int = 1              # Early exit layer for routing
    routing_threshold: float = 0.95  # Confidence threshold (0 = no routing)
```

---

## Preset Configurations

| Preset | Weights | Routing | Equivalent |
|--------|---------|---------|------------|
| `standard_llm` | L1:0, L2:0, L3:1 | disabled | Standard Transformer |
| `lpt` | L1:1/3, L2:1/3, L3:1/3 | disabled | Layer-wise Progressive Training |
| `standard_routing` | L1:0.5, L2:0, L3:0.5 | threshold=0.95 | Auxiliary Loss Training |
| `asymmetric_best` | L1:0.7, L2:0, L3:0.3 | threshold=0.95 | Best performing model |
| `lpt_routing` | L1:1/3, L2:1/3, L3:1/3 | threshold=0.7 | LPT + Early Exit |
| `asymmetric_with_l2` | L1:0.7, L2:1, L3:0.3 | threshold=0.95 | All-layer loss (not recommended) |

---

## Usage

```python
from experiments.universal_trainer import UniversalConfig, UniversalTrainer, PRESETS

# Use preset
config = PRESETS['asymmetric_best']

# Or create custom config
config = UniversalConfig(
    layer_weights={1: 0.8, 2: 0, 3: 0.2},
    routing_threshold=0.95
)

trainer = UniversalTrainer(config, vocab_size=vocab_size, device='cpu')

# Training
loss = trainer.compute_loss(model, x, y)

# Evaluation
stats = trainer.evaluate(model, val_batches)
# Returns: {ppl, acc, shallow_ratio, compute_cost}
```

---

## How Each Method is Represented

### Standard LLM

```python
# 最終層のみにロス
weights = {1: 0, 2: 0, 3: 1}
routing_threshold = 0  # ルーティングなし

# 等価な損失関数:
Loss = L3_loss
```

### LPT (Layer-wise Progressive Training)

```python
# 全層に均等重み
weights = {1: 1/3, 2: 1/3, 3: 1/3}
routing_threshold = 0  # ルーティングなし

# 等価な損失関数:
Loss = (L1_loss + L2_loss + L3_loss) / 3
```

### Standard Routing (Auxiliary Loss)

```python
# L1とL3に同等の重み
weights = {1: 0.5, 2: 0, 3: 0.5}
routing_threshold = 0.95

# 等価な損失関数:
Loss = 0.5 * L1_loss + 0.5 * L3_loss
```

### Asymmetric (Best)

```python
# Shallow重視
weights = {1: 0.7, 2: 0, 3: 0.3}
routing_threshold = 0.95

# 等価な損失関数:
Loss = 0.7 * L1_loss + 0.3 * L3_loss
```

---

## Routing Behavior

```
routing_threshold = 0:
  全トークン → L1 → L2 → L3 → Output
  (Standard LLM と同等)

routing_threshold > 0:
  Easy tokens (conf >= threshold) → L1 → Output
  Hard tokens (conf < threshold)  → L1 → L2 → L3 → Output
```

---

## New Features (v2.0)

### Dynamic Alpha (動的α)

学習中にαを変化させるCurriculum Learning的な手法:

```python
from experiments.universal_trainer import AlphaSchedule

# Linear decay: 0.9 -> 0.5
schedule = AlphaSchedule('linear', start=0.9, end=0.5)

# Cosine annealing
schedule = AlphaSchedule('cosine', start=0.9, end=0.5)

config = UniversalConfig(
    layer_weights={1: 0.9, 2: 0, 3: 0.1},
    routing_threshold=0.95,
    alpha_schedule=schedule
)
```

### Layer-wise Learning Rate

層ごとに異なる学習率を設定:

```python
config = UniversalConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},
    routing_threshold=0.95,
    layer_lr_scales={1: 1.0, 2: 0.5, 3: 0.1}  # L1 fast, L3 slow
)

trainer = UniversalTrainer(config, vocab_size=vocab_size)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)
```

**実験結果** (Layer-wise LR):
| Config | PPL | vs Baseline |
|--------|-----|-------------|
| Decreasing LR (1.0, 0.5, 0.1) | **18.52** | **19.3% 改善** |
| Increasing LR (0.1, 0.5, 1.0) | 21.14 | 7.9% 改善 |
| Standard (no Layer-wise LR) | 22.95 | baseline |

---

## Limitations

Universal Framework で**再現できる**もの:
- Standard LLM
- LPT
- Standard Routing
- Asymmetric Training
- 任意の層重み付け
- **動的α** ✅ (v2.0で追加)
- **Layer-wise Learning Rate** ✅ (v2.0で追加)

Universal Framework で**再現できない**もの:
1. **学習可能なConfidence Head**: 現在はmax(softmax)固定
2. **Multi-exit**: 複数のexit pointを持つアーキテクチャ
3. **Mixture of Experts**: トークンごとに異なるパスを選択
4. **知識蒸留**: Teacher-Student learning

詳細は [07_limitations.md](07_limitations.md) を参照。

---

## Results Verification

Universal Framework で以前の結果を再現:

| Model | Previous | Universal | Match |
|-------|----------|-----------|-------|
| Standard LLM | 34.86 | 34.86 | ✓ |
| LPT | 30.54 | 30.54 | ✓ |
| Standard Routing | 23.98 | 23.98 | ✓ |
| Asymmetric (α=0.7) | 22.95 | 22.95 | ✓ |
| LPT + Routing | 28.13 | 28.13 | ✓ |
