# Summary of All Results

## Final Ranking

| Rank | Model | PPL | Compute% | vs Standard 3L |
|------|-------|-----|----------|----------------|
| 🥇 | **Asymmetric (α=0.7, L2なし)** | **22.95** | 65.0% | **34.2% 改善** |
| 🥈 | Standard Routing (α=0.5) | 23.98 | 65.2% | 31.2% 改善 |
| 🥉 | LPT Routing | 28.13 | 46.6% | 19.3% 改善 |
| 4 | LPT (3L) | 30.54 | 100% | 12.4% 改善 |
| 5 | **Asymmetric+L2 (α=0.7)** | **32.07** | 42.5% | 8.0% 改善 |
| 6 | Standard (3L) | 34.86 | 100% | (baseline) |
| 7 | Standard (1L) | 35.29 | 33.3% | -1.2% |

---

## Best Practices

### 1. Training Method

| Use Case | Recommended Method |
|----------|-------------------|
| Standard Transformer | **LPT** (Layer-wise Progressive Training) |
| **Confidence-Routed** | **Asymmetric (α=0.7)** ⭐ |
| Alternative | Standard Routing (α=0.5) |

### 2. Architecture Choice

| Goal | Recommended | PPL | Compute |
|------|-------------|-----|---------|
| **Best quality** | **Asymmetric (α=0.7)** ⭐ | **22.95** | 65.0% |
| Second best | Standard Routing (α=0.5) | 23.98 | 65.2% |
| Best efficiency | LPT Routing | 28.13 | 46.6% |
| Simple & good | LPT (3L) | 30.54 | 100% |
| Memory constraints | Standard (1L) | 35.29 | 33.3% |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| **Asymmetric (α=0.7, L2なし) vs Standard 3L** | **34.2% 改善, 35.0% 計算削減** ⭐ |
| Standard Routing vs Standard 3L | 31.2% 改善, 34.8% 計算削減 |
| LPT Routing vs Standard 3L | 19.3% 改善, 53.4% 計算削減 |
| LPT vs Standard | 12.4% 改善 |
| Asymmetric vs Standard Routing | **4.3% さらに改善** |
| **L2ロス追加の影響** | **39.8% 悪化 (22.95 → 32.07)** ⚠️ |

---

## Key Insights

### 1. LPT vs Standard (for basic transformer)
- LPTは12.4%改善
- 各層に出力能力を持たせることで深い層も効果的に学習

### 2. Routing (for efficiency)
- 31.2%改善 + 34.8%計算削減
- 簡単なトークンはL1、難しいトークンはL3で処理

### 3. Asymmetric (α=0.7, L2なし) (best overall)
- 34.2%改善
- Shallow (L1) を重点的に訓練することで最高性能
- 多くのトークンは「簡単」なのでL1の精度向上が効果的

### 4. L2ロスの影響 (重要発見)
- **L2にロスを追加すると39.8%性能悪化**
- L2が「最終出力を作る」ように学習してしまう
- L2は純粋な中間層として機能させるべき
- L2ロスなしの場合、L2はDeep pathの特徴抽出に専念

### 5. 数学的同等性
- L2にロスを適用しない場合、以下は同等:
  - Asymmetric LPT: `forward_all_layers()` でL1, L3のみロス
  - Asymmetric Standard: `forward_train()` でshallow, deepのみロス

---

## Recommended Configuration (Universal Framework)

```python
from experiments.universal_trainer import UniversalConfig, PRESETS

# For best quality (推奨) ⭐
config = PRESETS['asymmetric_best']
# Equivalent to:
# UniversalConfig(layer_weights={1: 0.7, 2: 0, 3: 0.3}, routing_threshold=0.95)

# Alternative (simpler to understand)
config = PRESETS['standard_routing']
# Equivalent to:
# UniversalConfig(layer_weights={1: 0.5, 2: 0, 3: 0.5}, routing_threshold=0.95)

# For best efficiency (speed-focused)
config = PRESETS['lpt_routing']
# Equivalent to:
# UniversalConfig(layer_weights={1: 1/3, 2: 1/3, 3: 1/3}, routing_threshold=0.7)

# For simplicity (no routing)
config = PRESETS['lpt']
# Equivalent to:
# UniversalConfig(layer_weights={1: 1/3, 2: 1/3, 3: 1/3}, routing_threshold=0)

# Custom configuration
config = UniversalConfig(
    layer_weights={1: 0.8, 2: 0, 3: 0.2},  # α=0.8
    routing_threshold=0.95
)
```

詳細は [06_universal_framework.md](06_universal_framework.md) を参照。

---

## Experimental Notes

### Early Stopping
All models converge in **1 epoch** with strict early stopping due to:
1. Quick learning of basic patterns
2. Overfitting on small validation set
3. Early stopping preserves generalization

### Validation Behavior
- Train PPL continues to decrease
- Val PPL increases after epoch 1
- This indicates overfitting, not underfitting

---

## Future Work

1. **Larger models**: Test if findings scale
2. **More data**: Reduce overfitting tendency
3. **Different tasks**: Verify routing helps across tasks
4. **α の最適化**: より細かい α 値の探索 (0.6, 0.75, 0.8 など)
5. **動的 α**: 訓練中に α を変化させる手法
6. **Multi-exit**: 複数の exit point を持つアーキテクチャ
