# Saturation予測の研究まとめ

## 概要

Transformer の Early Exit において、**Saturation（飽和）** は重要な概念である。
Saturation とは、ある層で top-1 予測が確定し、後続の層でも変わらない現象を指す。

本ドキュメントでは、既存研究と我々の delta ベースの実験結果をまとめる。

---

## 1. 既存研究

### 1.1 CALM (Confident Adaptive Language Modeling)

**出典**: [Google Research, NeurIPS 2022](https://arxiv.org/abs/2207.07061)

**概要**: 入力と生成ステップごとに異なる計算量を動的に割り当てるフレームワーク。

**Confidence Measure（3種類）**:

| 方式 | 説明 | 特徴 |
|------|------|------|
| Softmax Response | softmax の最大確率値 | シンプル、高速 |
| State Propagation | 連続層間の cosine similarity | FLOPS が軽い |
| Early-exit Classifier | 線形 classifier（\|d\|+1 パラメータ） | 専用訓練が必要 |

**結果**:
- 平均で 1/3〜1/2 の層数で full model と同等の性能
- 最大 3x のスピードアップ

**訓練方法**:
- 全層の予測の加重平均で損失を計算（上層ほど重み大）
- Early-exit classifier は frozen model の後に追加訓練

---

### 1.2 LayerSkip

**出典**: [Meta, ACL 2024](https://arxiv.org/abs/2404.16710)

**概要**: Layer Dropout + Early Exit + Self-Speculative Decoding の統合手法。

**特徴**:
- 訓練時: 浅い層は低 dropout率、深い層は高 dropout率
- 全層で同じ exit head を共有
- 推論時: early exit + self-speculative decoding で検証

**Saturation に関する知見**:
> "Even with a perfect predictor with zero compute overhead, we can only save up to 26% of computation"
> (GPT2 で平均 23.45/32 層が必要)

**問題点の指摘**:
- Early exit すると後続トークンの context が失われる
- → Skip（層をスキップ）の方が良い場合がある

---

### 1.3 Looking Beyond the Top-1

**出典**: [ICML 2025](https://openreview.net/forum?id=2B11W1Z6ID)

**概要**: Top-k トークンの saturation event を分析。

**重要な発見**:
1. **Sequential Saturation**: top-1, top-2, top-3... の順で saturation が起きる
2. **Task Transition Mechanism**: hidden layer に「次のタスク」がエンコードされている
3. **アーキテクチャ非依存**: 言語・視覚・音声モデルで共通の現象

**応用**:
- hidden layer embedding から現在のタスクを予測可能
- token-level early-exit 戦略に活用

---

### 1.4 EE-LLM

**出典**: [Alibaba, 2023](https://arxiv.org/pdf/2312.04916)

**概要**: 大規模 LLM の Early Exit 訓練・推論フレームワーク。

**特徴**:
- 3D parallelism（Data/Tensor/Pipeline）対応
- 訓練時: 全層で exit loss を計算
- 推論時のみ early exit を適用

---

### 1.5 既存手法の共通点

| 手法 | Exit 判定の入力 | 予測対象 |
|------|----------------|----------|
| CALM | hidden states | confidence / consistency |
| LayerSkip | hidden states | early exit 可否 |
| Looking Beyond Top-1 | hidden states | task transition |
| EE-LLM | hidden states + logits | exit 可否 |

**共通点**: 全て **hidden states** を直接使用している。

---

## 2. Delta ベースのアプローチ（本研究）

### 2.1 着想

Transformer の残差接続に注目：

```
h_next = h + Attention(h) + FFN(h)
       = h + delta
```

**仮説**: delta（残差）が小さければ、top-1 は変わらない → early exit 可能

### 2.2 既存手法との差異

| 観点 | 既存手法 | 本アプローチ |
|------|----------|--------------|
| 入力 | hidden states (h) | delta (h_out - h_in) |
| 計算タイミング | 層の出力後 | 層の出力後（deltaは副産物） |
| 追加計算 | なし〜少量 | MLP のみ |
| 新規性 | - | deltaベースは既存研究にない |

---

## 3. 実験結果

### 3.1 実験設定

- **モデル**: dim=64, num_heads=4, ffn_dim=256, num_layers=8
- **データ**: WikiText, 65,536 tokens
- **評価指標**: F1 score（saturation 検出）

### 3.2 Delta Norm vs Top-1 変化の相関

| 遷移 | 変化率 | 相関係数 r |
|------|--------|-----------|
| Layer 1→2 | 60.8% | 0.216 |
| Layer 2→3 | 66.0% | 0.132 |
| Layer 3→4 | 60.8% | 0.141 |
| Layer 4→5 | 56.1% | 0.135 |
| Layer 5→6 | 48.3% | 0.154 |
| Layer 6→7 | 39.3% | 0.189 |
| Layer 7→8 | 46.2% | 0.154 |

**観察**:
- 深い層ほど変化率が減少（saturation 増加）
- Layer 6→7 で変化率が最小（39.3%）= saturation 最大
- 相関は弱い（r ≈ 0.15）が有意

### 3.3 Saturation 検出 F1（手法比較）

| 遷移 | Delta Norm | Linear | MLP |
|------|------------|--------|-----|
| 1→2 | 58.3% | 48.1% | **65.8%** |
| 2→3 | 50.8% | 48.5% | **57.4%** |
| 3→4 | 56.2% | 48.3% | **60.7%** |
| 4→5 | 61.1% | 51.2% | **63.7%** |
| 5→6 | 68.7% | 67.0% | **74.6%** |
| 6→7 | 74.2% | 76.0% | **79.2%** |
| 7→8 | 70.1% | 71.9% | **77.9%** |
| **平均** | 62.8% | 58.7% | **68.5%** |

**結論**:
- **MLP が全層で最良**（平均 F1 = 68.5%）
- Delta Norm threshold より **+5.7%** 改善
- Linear は Norm より悪い（非線形性が必要）

### 3.4 Delta 統計（層別）

| Layer | \|delta\| mean | \|delta\|/\|h_in\| |
|-------|----------------|-------------------|
| 1 | 3.30 | 0.419 |
| 2 | 3.31 | 0.409 |
| 3 | 3.25 | 0.402 |
| 4 | 3.27 | 0.405 |
| 5 | 3.35 | 0.415 |
| 6 | 3.28 | 0.405 |
| 7 | 3.37 | 0.417 |
| 8 | **6.43** | **0.796** |

**観察**:
- Layer 1-7: delta は安定（約 3.3）
- Layer 8: delta が約 2倍（最終層は出力特化の変換）

---

## 4. 考察

### 4.1 Delta ベースの利点

1. **追加計算が少ない**: delta は forward 中に自然に計算される
2. **方向情報を活用**: norm だけでなくベクトル全体を使える
3. **層の変化量を直接観測**: hidden states よりも変化に特化

### 4.2 現在の限界

1. **F1 = 68.5%** は CALM の報告値（〜90%）より低い
   - ただしモデルサイズ・データセットが異なる
2. **最終層は特殊**: delta が大きく、別の扱いが必要
3. **浅い層での性能が低い**: Layer 1-4 は F1 < 65%

### 4.3 今後の方向性

1. **Delta + Hidden States の組み合わせ**
   - 入力: `[h, delta]` の concat
   - より多くの情報を活用

2. **層ごとの専用 MLP**
   - 各層の特性に合わせた predictor

3. **大規模モデルでの検証**
   - dim=256 以上でのスケーリング

---

## 5. 参考文献

1. Schuster et al. (2022). [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061). NeurIPS 2022.
2. Elhoushi et al. (2024). [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710). ACL 2024.
3. Lioubashevski et al. (2025). [Looking Beyond the Top-1: Transformers Determine Top Tokens in Order](https://openreview.net/forum?id=2B11W1Z6ID). ICML 2025.
4. Chen et al. (2023). [EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models](https://arxiv.org/pdf/2312.04916).
5. Geva et al. (2022). Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. EMNLP 2022.
