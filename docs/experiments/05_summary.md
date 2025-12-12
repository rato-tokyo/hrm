# Summary of All Results

**⚠️ HISTORICAL DOCUMENT**: This document describes experiments conducted with the older framework that included `layer_lr_scales` as a core option. The current LEGO framework (v0.2.0) uses only **2 core options** (`layer_weights` and `routing_threshold`). See [CLAUDE.md](../../CLAUDE.md) for the latest framework specification.

---

## Terminology Mapping

| This Project (旧) | Academic Term | Reference |
|-------------------|---------------|-----------|
| LPT | **Deep Supervision** | Lee et al., 2015 |
| Standard Routing | **Auxiliary Loss Training** | Elbayad et al., 2020 |
| Confidence-based Routing | **Early Exit** | Teerapittayanon et al., 2016 |
| Layer-wise Learning Rate | **Discriminative Fine-Tuning** | Howard & Ruder, 2018 |

詳細は [REFERENCES.md](../REFERENCES.md) を参照。

---

## Final Ranking

| Rank | Model | PPL | Compute% | vs Standard 3L |
|------|-------|-----|----------|----------------|
| 🥇 | **Discriminative Fine-Tuning (Decreasing LR)** | **18.52** | 65.2% | **46.9% 改善** |
| 🥈 | Discriminative Fine-Tuning (Increasing LR) | 21.14 | 72.1% | 39.3% 改善 |
| 🥉 | Asymmetric Auxiliary Loss (α=0.8) | 22.40 | 65.2% | 35.7% 改善 |
| 4 | Asymmetric Auxiliary Loss (α=0.7) | 22.95 | 65.0% | 34.2% 改善 |
| 5 | Auxiliary Loss Training (α=0.5) | 23.98 | 65.2% | 31.2% 改善 |
| 6 | Deep Supervision + Early Exit | 28.13 | 46.6% | 19.3% 改善 |
| 7 | Deep Supervision (3L) | 30.54 | 100% | 12.4% 改善 |
| 8 | Standard (3L) | 34.86 | 100% | (baseline) |

---

## Best Practices

### 1. Training Method

| Use Case | Recommended Method | Reference |
|----------|-------------------|-----------|
| Standard Transformer | **Deep Supervision** | Lee et al., 2015 |
| **Early Exit (Best Quality)** | **Asymmetric Auxiliary Loss (α=0.7) + Discriminative Fine-Tuning** ⭐ | Ours |
| Early Exit (Alternative) | Auxiliary Loss Training (α=0.5) | Elbayad et al., 2020 |

### 2. Architecture Choice

| Goal | Recommended | PPL | Compute |
|------|-------------|-----|---------|
| **Best quality** | **Discriminative Fine-Tuning (Decreasing)** ⭐ | **18.52** | 65.2% |
| Second best | Asymmetric Auxiliary Loss (α=0.7) | 22.95 | 65.0% |
| Best efficiency | Deep Supervision + Early Exit | 28.13 | 46.6% |
| Simple & good | Deep Supervision (3L) | 30.54 | 100% |
| Memory constraints | Standard (1L) | 35.29 | 33.3% |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| **Discriminative Fine-Tuning vs Standard 3L** | **46.9% 改善** ⭐ |
| Asymmetric Auxiliary Loss (α=0.7) vs Standard 3L | 34.2% 改善, 35.0% 計算削減 |
| Auxiliary Loss Training vs Standard 3L | 31.2% 改善, 34.8% 計算削減 |
| Deep Supervision + Early Exit vs Standard 3L | 19.3% 改善, 53.4% 計算削減 |
| Deep Supervision vs Standard | 12.4% 改善 |
| **L2ロス追加の影響** | **39.8% 悪化 (22.95 → 32.07)** ⚠️ |

---

## Key Insights

### 1. Deep Supervision vs Standard (for basic transformer)
- Deep Supervision は12.4%改善
- 各層に出力能力を持たせることで深い層も効果的に学習

### 2. Early Exit (for efficiency)
- 31.2%改善 + 34.8%計算削減
- 簡単なトークンはL1、難しいトークンはL3で処理

### 3. Asymmetric Auxiliary Loss (α=0.7, L2ロスなし)
- 34.2%改善
- Shallow (L1) を重点的に訓練することで高性能
- 多くのトークンは「簡単」なのでL1の精度向上が効果的

### 4. Discriminative Fine-Tuning (best overall) ⭐
- **46.9%改善（最良結果）**
- 浅い層に高い学習率、深い層に低い学習率
- ULMFiT (Howard & Ruder, 2018) で提案された手法をEarly Exitに適用

### 5. L2ロスの影響 (重要発見)
- **L2にロスを追加すると39.8%性能悪化**
- L2が「最終出力を作る」ように学習してしまう
- L2は純粋な中間層として機能させるべき
- L2ロスなしの場合、L2はDeep pathの特徴抽出に専念

### 6. 数学的同等性
- L2にロスを適用しない場合、以下は同等:
  - `forward_all_layers()` でL1, L3のみロス
  - `forward_train()` でshallow, deepのみロス

---

## Recommended Configuration (Universal Framework)

```python
from experiments.universal_trainer import UniversalConfig, PRESETS

# For best quality (推奨) ⭐
config = UniversalConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},
    routing_threshold=0.95,
    layer_lr_scales={1: 1.0, 2: 0.5, 3: 0.1}  # Discriminative Fine-Tuning
)

# Alternative (simpler to understand)
config = PRESETS['auxiliary_loss']
# Equivalent to:
# UniversalConfig(layer_weights={1: 0.5, 2: 0, 3: 0.5}, routing_threshold=0.95)

# For best efficiency (speed-focused)
config = PRESETS['deep_supervision_routing']
# Equivalent to:
# UniversalConfig(layer_weights={1: 1/3, 2: 1/3, 3: 1/3}, routing_threshold=0.7)

# For simplicity (no routing)
config = PRESETS['deep_supervision']
# Equivalent to:
# UniversalConfig(layer_weights={1: 1/3, 2: 1/3, 3: 1/3}, routing_threshold=0)
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
5. **Multi-exit**: 複数の exit point を持つアーキテクチャ
6. **学習可能なConfidence Head**: max(softmax)以外の手法

## References

- Lee, C.-Y., et al. (2015). **Deeply-Supervised Nets**. AISTATS 2015. https://arxiv.org/abs/1409.5185
- Howard, J., & Ruder, S. (2018). **Universal Language Model Fine-tuning for Text Classification**. ACL 2018. https://arxiv.org/abs/1801.06146
- Elbayad, M., et al. (2020). **Depth-Adaptive Transformer**. ICLR 2020. https://arxiv.org/abs/1910.10073
- Teerapittayanon, S., et al. (2016). **BranchyNet: Fast Inference via Early Exiting**. ICPR 2016. https://arxiv.org/abs/1709.01686
