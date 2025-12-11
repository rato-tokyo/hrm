# Experiment 5 & 6: Asymmetric Auxiliary Loss Training

**注意**: このドキュメントは特定の実験設定での結果を記録したものです。
EASEフレームワークの設定は自由であり、ここに記載された設定が最適とは限りません。
フレームワークの概要は [06_universal_framework.md](06_universal_framework.md) を参照してください。

---

## Goal

L1（Shallow）とL3（Deep）に異なる重みでロスを適用した場合の効果を検証する。

**Base Reference**: Elbayad et al. (2020) "Depth-Adaptive Transformer"

## 実験設定

- モデル: 3層 Transformer
- データ: WikiText-2 (200K chars)
- 損失: Cross Entropy

---

## Architecture Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│              設定A: 中間層ロスなし                                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input x                                                                │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 1   │ → Output → Loss₁ (重みα)                              │
│  └─────────────┘                                                       │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 2   │ (ロスなし)                                             │
│  └─────────────┘                                                       │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 3   │ → Output → Loss₃ (重み1-α)                            │
│  └─────────────┘                                                       │
│                                                                         │
│  Total Loss = α * Loss₁ + (1-α) * Loss₃                                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│              設定B: 全層ロス (Deep Supervision)                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input x                                                                │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 1   │ → Output → Loss₁ (重みα)                              │
│  └─────────────┘                                                       │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 2   │ → Output → Loss₂ (重み1.0)                            │
│  └─────────────┘                                                       │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 3   │ → Output → Loss₃ (重み1-α)                            │
│  └─────────────┘                                                       │
│                                                                         │
│  Total Loss = α * Loss₁ + Loss₂ + (1-α) * Loss₃                        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Experiment 5: 設定A（中間層ロスなし）

### Training Method

```python
all_outputs = model.forward_all_layers(x)
shallow_out = all_outputs[0]   # L1
deep_out = all_outputs[-1]     # L3

shallow_loss = cross_entropy(shallow_out, target)
deep_loss = cross_entropy(deep_out, target)
total_loss = alpha * shallow_loss + (1 - alpha) * deep_loss
```

### Results

| α | L1重み | L3重み | PPL | Shallow% | Compute% |
|---|--------|--------|-----|----------|----------|
| 0.3 | 30% | 70% | 24.88 | 72.0% | 52.0% |
| 0.5 | 50% | 50% | 23.98 | 52.3% | 65.2% |
| 0.7 | 70% | 30% | 22.95 | 52.4% | 65.0% |

### Threshold Analysis (α=0.7)

| Threshold | PPL | Shallow% | Compute% |
|-----------|-----|----------|----------|
| 0.50 | 41.62 | 94.4% | 37.0% |
| 0.70 | 40.48 | 87.8% | 41.4% |
| 0.80 | 40.57 | 84.5% | 43.6% |
| 0.90 | 36.94 | 73.4% | 51.1% |
| 0.95 | 31.46 | 59.7% | 60.2% |

---

## Experiment 6: 設定B（全層ロス）

### Training Method

```python
all_outputs = model.forward_all_layers(x)
shallow_out = all_outputs[0]   # L1
middle_out = all_outputs[1]    # L2
deep_out = all_outputs[-1]     # L3

shallow_loss = cross_entropy(shallow_out, target)
middle_loss = cross_entropy(middle_out, target)
deep_loss = cross_entropy(deep_out, target)
total_loss = alpha * shallow_loss + middle_loss + (1 - alpha) * deep_loss
```

### Results (α=0.7)

| 設定 | PPL | Shallow% | Compute% |
|------|-----|----------|----------|
| 設定A (中間層ロスなし) | 22.95 | 52.4% | 65.0% |
| 設定B (全層ロス) | 32.07 | 86.3% | 42.5% |

この実験設定では、設定Aの方がPPLが良い結果となった。

### Threshold Analysis Comparison

| 設定 | Threshold | PPL | Shallow% | Compute% |
|------|-----------|-----|----------|----------|
| 設定A | 0.70 | 40.48 | 87.8% | 41.4% |
| 設定B | 0.70 | 50.24 | 91.0% | 39.4% |
| 設定A | 0.80 | 40.57 | 84.5% | 43.6% |
| 設定B | 0.80 | 51.46 | 87.5% | 41.7% |
| 設定A | 0.90 | 36.94 | 73.4% | 51.1% |
| 設定B | 0.90 | 53.10 | 79.0% | 47.4% |
| 設定A | 0.95 | 31.46 | 59.7% | 60.2% |
| 設定B | 0.95 | 54.90 | 68.9% | 54.0% |

---

## 実験結果の考察

この特定の実験設定（3層、WikiText-2、200K chars）では以下の傾向が観察された：

1. **設定Aが設定Bより良いPPLを示した**
   - ただし、これはこの実験設定に特有の結果である可能性がある
   - 異なるモデルサイズ、データセット、タスクでは異なる結果になりうる

2. **αの値による違い**
   - この実験ではα=0.7が最良だったが、これも実験設定に依存する

3. **Thresholdの影響**
   - 高いthreshold（0.95）で良いPPLを示したが、これもタスク依存

---

## 注意事項

**重要**: この実験結果を一般化して「中間層の損失は0にすべき」「α=0.7が最適」と結論づけることはできません。

最適な設定は以下に依存します：
- モデルの層数と次元
- データセットの種類とサイズ
- タスクの性質
- 計算リソースの制約

EASEフレームワークを使用する際は、自身のユースケースに合わせて実験を行い、最適な設定を見つけてください。

---

## References

- Elbayad, M., Gu, J., Grave, E., & Auli, M. (2020). **Depth-Adaptive Transformer**. ICLR 2020. https://arxiv.org/abs/1910.10073
- Lee, C.-Y., et al. (2015). **Deeply-Supervised Nets**. AISTATS 2015. https://arxiv.org/abs/1409.5185
