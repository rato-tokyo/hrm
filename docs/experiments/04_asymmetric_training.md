# Experiment 5 & 6: Asymmetric Auxiliary Loss Training

## Goal

L1（Shallow）とL3（Deep）に異なる重みでロスを適用し、各パスを専門化させる。

**Base Reference**: Elbayad et al. (2020) "Depth-Adaptive Transformer"

**Our Contribution**:
1. 非対称な重み付け（α≠0.5）の効果を検証
2. 中間層（L2）の損失を0にする重要性を発見

## Architecture Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│              Asymmetric Auxiliary Loss (L2ロスなし) - 推奨               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input x                                                                │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 1   │ → Output → Loss₁ (Shallow専用, 重みα)                 │
│  └─────────────┘                                                       │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 2   │ (中間層, ロスなし → 純粋な特徴抽出)                     │
│  └─────────────┘                                                       │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 3   │ → Output → Loss₃ (Deep専用, 重み1-α)                  │
│  └─────────────┘                                                       │
│                                                                         │
│  Total Loss = α * Loss₁ + (1-α) * Loss₃                                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│              Deep Supervision (全層ロス) - 非推奨                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input x                                                                │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 1   │ → Output → Loss₁ (Shallow専用, 重みα)                 │
│  └─────────────┘                                                       │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 2   │ → Output → Loss₂ (重み1.0)                            │
│  └─────────────┘                                                       │
│     ↓                                                                   │
│  ┌─────────────┐                                                       │
│  │   Layer 3   │ → Output → Loss₃ (Deep専用, 重み1-α)                  │
│  └─────────────┘                                                       │
│                                                                         │
│  Total Loss = α * Loss₁ + Loss₂ + (1-α) * Loss₃                        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Experiment 5: Asymmetric Auxiliary Loss (L2ロスなし)

### Training Method

L1とL3の出力を `forward_all_layers()` から取得：

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
| **0.7** | **70%** | **30%** | **22.95** | **52.4%** | **65.0%** |

### Threshold Analysis (Best: α=0.7)

| Threshold | PPL | Shallow% | Compute% |
|-----------|-----|----------|----------|
| 0.50 | 41.62 | 94.4% | 37.0% |
| 0.70 | 40.48 | 87.8% | 41.4% |
| 0.80 | 40.57 | 84.5% | 43.6% |
| 0.90 | 36.94 | 73.4% | 51.1% |
| **0.95** | **31.46** | **59.7%** | **60.2%** |

---

## Experiment 6: Deep Supervision with Asymmetric Weights (全層ロス) - 公平比較

### Training Method

L1, L2, L3 すべてにロスを適用：

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

### Results (α=0.7 での公平比較)

| Model | PPL | Shallow% | Compute% |
|-------|-----|----------|----------|
| **Asymmetric Auxiliary Loss (L2なし)** | **22.95** | 52.4% | 65.0% |
| Deep Supervision + Asymmetric (全層) | 32.07 | 86.3% | 42.5% |

**L2ロス追加の影響: PPL +9.13 (+39.8%) 悪化**

### Threshold Analysis Comparison

| Config | Threshold | PPL | Shallow% | Compute% |
|--------|-----------|-----|----------|----------|
| Asymmetric Auxiliary Loss (L2なし) | 0.70 | 40.48 | 87.8% | 41.4% |
| Deep Supervision + Asymmetric | 0.70 | 50.24 | 91.0% | 39.4% |
| Asymmetric Auxiliary Loss (L2なし) | 0.80 | 40.57 | 84.5% | 43.6% |
| Deep Supervision + Asymmetric | 0.80 | 51.46 | 87.5% | 41.7% |
| Asymmetric Auxiliary Loss (L2なし) | 0.90 | 36.94 | 73.4% | 51.1% |
| Deep Supervision + Asymmetric | 0.90 | 53.10 | 79.0% | 47.4% |
| **Asymmetric Auxiliary Loss (L2なし)** | **0.95** | **31.46** | 59.7% | 60.2% |
| Deep Supervision + Asymmetric | 0.95 | 54.90 | 68.9% | 54.0% |

---

## 重要な発見: 中間層（L2）の損失を0にすべき

### なぜL2ロスを追加すると性能が悪化するか

```
L2ロスあり (Deep Supervision):
  L2 が「最終出力を作る」ように学習
  ↓
  L2 の出力が L3 への良い中間表現を生成できなくなる
  ↓
  L1 の confidence が過度に高くなる (86.3%)
  ↓
  実際には難しいトークンも shallow で処理してしまう
  ↓
  PPL 悪化

L2ロスなし (Asymmetric Auxiliary Loss):
  L2 は純粋な中間層として機能
  ↓
  L1 → L2 → L3 の Deep path が適切に特徴抽出
  ↓
  L1 は適切な confidence を維持 (52.4%)
  ↓
  難しいトークンは Deep path で正しく処理
  ↓
  最良の PPL
```

### 数学的同等性について

L2ロスを適用しない場合、以下は数学的に同等：
- `forward_all_layers()` を使用して L1, L3 のみにロス適用
- `forward_train()` を使用して shallow, deep にロス適用

どちらも: `Loss = α * L1_loss + (1-α) * L3_loss`

---

## Key Findings

1. **Asymmetric Auxiliary Loss (α=0.7, L2ロスなし) が全手法で最良** (PPL: 22.95)
2. **Standard 3L より 34.2% 改善、計算コスト 35.0% 削減**
3. **Standard Auxiliary Loss (23.98) より 4.3% さらに改善**
4. **Shallow重視 (α=0.7) が最も効果的**
5. **L2ロスを追加すると39.8%性能悪化** (22.95 → 32.07)
6. **L2は純粋な中間層として機能させるべき**

---

## αの解釈

| α値 | 意味 | 結果 |
|-----|------|------|
| 0.3 | Deep重視 | PPL 24.88 |
| 0.5 | バランス（= Standard Auxiliary Loss） | PPL 23.98 |
| **0.7** | **Shallow重視** | **PPL 22.95 (最良)** |

### Shallow重視が最良の理由

- 多くのトークンは「簡単」→ L1 で処理可能
- L1 の精度向上が全体の PPL を大幅に改善
- 難しいトークンは少数 → L3 の重みは低くても十分

---

## Why Asymmetric Auxiliary Loss Works Best

```
Standard Auxiliary Loss (α=0.5):
  Loss = 0.5 * L1_loss + 0.5 * L3_loss
  → L1とL3が同等の重みで学習

Asymmetric Auxiliary Loss (α=0.7):
  Loss = 0.7 * L1_loss + 0.3 * L3_loss
  → L1 (Shallow) に重点を置く
  → L2 は純粋な中間層として機能
  → Shallow path がより専門化される
  → 簡単なトークンの処理精度が向上
```

---

## Recommended Configuration

```python
# Best configuration
config = {
    'architecture': 'Early Exit (Confidence-Routed)',
    'training': 'Asymmetric Auxiliary Loss',
    'alpha': 0.7,  # Shallow重視
    'exit_layer': 1,
    'threshold': 0.95,  # Quality-focused
}
```

## References

- Elbayad, M., Gu, J., Grave, E., & Auli, M. (2020). **Depth-Adaptive Transformer**. ICLR 2020. https://arxiv.org/abs/1910.10073
- Lee, C.-Y., et al. (2015). **Deeply-Supervised Nets**. AISTATS 2015. https://arxiv.org/abs/1409.5185
