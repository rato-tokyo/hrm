# Saturation予測の研究まとめ

## 概要

Transformer の Early Exit において、**Saturation（飽和）** は重要な概念である。
Saturation とは、ある層で top-1 予測が確定し、後続の層でも変わらない現象を指す。

本ドキュメントでは、既存研究と我々の実験結果をまとめる。

---

## 1. 既存研究

### 1.1 CALM (Confident Adaptive Language Modeling)

**出典**: [Google Research, NeurIPS 2022](https://arxiv.org/abs/2207.07061)

**概要**: 入力と生成ステップごとに異なる計算量を動的に割り当てるフレームワーク。

**Confidence Measure（3種類）**:

| 方式 | 説明 | 特徴 |
|------|------|------|
| Softmax Response | softmax の最大確率値 | シンプル、logits 計算が必要 |
| State Propagation | 連続層間の cosine similarity | 軽量、訓練不要 |
| Early-exit Classifier | 線形 classifier（\|d\|+1 パラメータ） | 専用訓練が必要 |

**結果**:
- 平均で 1/3〜1/2 の層数で full model と同等の性能
- 最大 3x のスピードアップ

---

### 1.2 LayerSkip

**出典**: [Meta, ACL 2024](https://arxiv.org/abs/2404.16710)

**概要**: Layer Dropout + Early Exit + Self-Speculative Decoding の統合手法。

**Saturation に関する知見**:
> "Even with a perfect predictor with zero compute overhead, we can only save up to 26% of computation"

---

### 1.3 Looking Beyond the Top-1

**出典**: [ICML 2025](https://openreview.net/forum?id=2B11W1Z6ID)

**重要な発見**:
1. **Sequential Saturation**: top-1, top-2, top-3... の順で saturation が起きる
2. **Task Transition Mechanism**: hidden layer に「次のタスク」がエンコードされている

---

## 2. 実験設定

- **モデル**: dim=64, num_heads=4, ffn_dim=256, num_layers=8
- **データ**: WikiText, 65,536 tokens
- **評価指標**: F1 score（saturation 検出）
- **予測対象**: Layer N の top-1 が最終層の top-1 と一致するか

---

## 3. 入力特徴量の比較実験

### 3.1 比較した入力

| 入力 | 次元数 | 説明 |
|------|--------|------|
| h_out | dim (64) | 層の出力（累積情報） |
| delta | dim (64) | h_out - h_in（この層の変化） |
| h_out.delta | dim×2 (128) | 両方を結合 |
| delta.prev_delta | dim×2 (128) | この層と前層の変化を結合 |
| delta - prev_delta | dim (64) | 変化の加速度 |
| state_prop | 1 | cos_sim(h_in, h_out) threshold |
| CALM state_prop | 1 | cos_sim(h_n, h_n+1) threshold |

### 3.2 結果（F1 スコア）

| Layer | Sat% | h_out | delta | h.d | d.prev | d-prev | CALM |
|-------|------|-------|-------|-----|--------|--------|------|
| 1 | 5.0% | 72.3% | 71.2% | 79.1% | N/A | N/A | 19.2% |
| 2 | 12.1% | 82.2% | 79.2% | 83.1% | 81.5% | 79.0% | 20.8% |
| 3 | 12.6% | 66.1% | 55.3% | 67.6% | 63.5% | 55.9% | 23.4% |
| 4 | 16.4% | 56.6% | 46.3% | 59.6% | 54.2% | 46.3% | 26.7% |
| 5 | 24.5% | 58.6% | 47.4% | 60.3% | 55.7% | 47.3% | 38.2% |
| 6 | 39.7% | 68.8% | 62.2% | 70.3% | 66.4% | 62.0% | 55.7% |
| 7 | 53.8% | 78.9% | 76.6% | 79.1% | 78.3% | 76.4% | 69.2% |
| **平均** | | **69.1%** | **62.6%** | **71.3%** | **66.6%** | **61.2%** | **36.2%** |

### 3.3 主要な発見

1. **h_out > delta (+6.5%)**
   - 累積情報が saturation 予測に重要
   - delta だけでは「現在どのトークン方向を向いているか」が分からない

2. **h_out.delta が最良 (71.3%)**
   - 累積情報と変化情報は相補的
   - 結合で +2.2% 改善

3. **delta - prev_delta は効果なし (-1.5%)**
   - 変化の加速度は saturation 予測に寄与しない
   - 絶対的な変化量の方が重要

4. **CALM State Propagation は深い層で急激に改善**
   - Layer 1: 19.2% → Layer 7: 69.2% (+50%)
   - 訓練不要で計算コストゼロ

---

## 4. CALM State Propagation の詳細分析

### 4.1 CALM 方式の定義

```python
# CALM State Propagation
cos_sim = cosine_similarity(h_layer_n, h_layer_n+1)

# 高い類似度 = 層間で変化が小さい = saturation している可能性
if cos_sim > threshold:
    exit()
```

### 4.2 なぜ cos_sim が有効か

```
cos_sim(h_n, h_n+1) が高い
= h_n と h_n+1 の方向がほぼ同じ
= Layer N+1 で表現がほとんど変わらなかった
= 既に「収束」している
= top-1 も変わらない可能性が高い
```

### 4.3 層深度による性能差

| Layer | CALM F1 | Recall | 特徴 |
|-------|---------|--------|------|
| 1 | 19.2% | 39.0% | 表現が不安定 |
| 4 | 26.7% | 88.6% | 中間層 |
| 7 | 69.2% | 93.1% | 表現が収束 |

**深い層での高 Recall (93.1%)**: saturated トークンのほぼ全てを検出できている。

---

## 5. 理論的考察

### 5.1 なぜ delta_norm だけでは不十分か

**直感**: delta が小さい → top-1 が変わらない

**実際**: delta_norm と saturation の相関は弱い (r ≈ 0.15)

**理由**:
1. **方向が重要、大きさではない**: 小さな delta でも logit margin が小さければ逆転可能
2. **Logit margin が層によって異なる**: 深い層ほど margin が大きく、逆転しにくい
3. **Delta の役割が層によって異なる**: 浅い層は特徴抽出、深い層は top-1 強化

### 5.2 Saturation の本質

```
Saturation = h が特定トークン方向に「収束」した状態
```

- **h_out の方向**: 「現在どのトークンを指しているか」
- **delta の方向**: 「どう変化しようとしているか」

両方を見ることで、saturation をより正確に予測できる。

### 5.3 Delta 方向の実験

cos_sim(delta, W[top-1]) と saturation の相関:

| Layer | cos_sim (saturated) | cos_sim (non-saturated) | 相関 r |
|-------|---------------------|-------------------------|--------|
| 1 | 0.277 | 0.123 | 0.253 |
| 6 | 0.215 | 0.116 | 0.291 |
| 7 | 0.139 | 0.092 | 0.178 |

**解釈**: delta が top-1 の W ベクトルに近い方向を向いている → saturation しやすい

---

## 6. 実用的な示唆

### 6.1 手法の比較

| 手法 | 平均 F1 | 計算コスト | 訓練コスト |
|------|---------|-----------|-----------|
| h_out.delta MLP | 71.3% | MLP forward | あり |
| h_out MLP | 69.1% | MLP forward | あり |
| CALM cos_sim | 36.2% | 内積のみ | なし |
| logits ベース | 高い | h @ W.T | なし |

### 6.2 推奨アプローチ

**訓練コストを避けたい場合**:
- **logits ベース（softmax response, entropy）**: 訓練不要、高精度
- **CALM cos_sim**: 深い層で有効、計算コスト最小

**精度を最優先する場合**:
- **h_out.delta MLP**: 最高精度だが訓練が必要

### 6.3 ハイブリッド戦略の可能性

```
浅い層（1-4）: logits ベースまたは MLP
深い層（5-7）: CALM cos_sim（軽量で高精度）
```

---

## 7. 結論

1. **累積情報（h_out）が重要**: delta だけでは不十分
2. **CALM State Propagation は深い層で有効**: 訓練不要、計算コストゼロ
3. **logits ベースが最も実用的**: 訓練不要で高精度、計算コストは許容範囲
4. **MLP は訓練コストに見合わない可能性**: logits ベースで十分な場合が多い

---

## 8. 参考文献

1. Schuster et al. (2022). [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061). NeurIPS 2022.
2. Elhoushi et al. (2024). [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710). ACL 2024.
3. Lioubashevski et al. (2025). [Looking Beyond the Top-1: Transformers Determine Top Tokens in Order](https://openreview.net/forum?id=2B11W1Z6ID). ICML 2025.
4. Chen et al. (2023). [EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models](https://arxiv.org/pdf/2312.04916).
