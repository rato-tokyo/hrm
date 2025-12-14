# Saturation Event と Low-Rank 近似の実験結果

## 概要

2024-12-15に実施した、TransformerにおけるSaturation Event（予測確定現象）と出力層の低ランク近似に関する実験結果をまとめる。

---

## 背景と仮説

### Saturation Event とは

先行研究（Geva et al., 2022; CALM）で発見された現象：
> 「モデルの最終予測（top-1トークン）は、ある浅い層で確定し、その後の深い層でも変わらない」

この現象は：
- Transformerアーキテクチャに固有
- 訓練されていないモデルでも発生
- 言語・画像・音声モデル全てで確認

### 仮説

1. **Saturation予測**: hidden statesから「top-1が変わるか」を予測できれば、早期exitの判断に使える
2. **低ランク近似**: 出力層 W を低ランク分解すれば、top-1予測を高速化できる
3. **大規模モデルでの効果**: dimが大きいほど低ランク近似の恩恵が大きい

---

## 実験設定

### モデル構成

| 設定 | 小モデル | 大モデル |
|------|----------|----------|
| dim | 64 | 256 |
| num_heads | 4 | 8 |
| ffn_dim | 256 | 1024 |
| num_layers | 4 | 4 |
| パラメータ数 | ~0.5M | ~10M |

### データ
- WikiText-2
- 65,536 トークン（validation set）
- vocab_size ≈ 70,000

---

## 実験1: Saturation Rate

**定義**: Layer i の top-1 予測が最終層（Layer 4）と一致する割合

### 結果

| 層の遷移 | dim=64 | dim=256 |
|----------|--------|---------|
| Layer 1→4 | 9.5% | 10.7% |
| Layer 2→4 | 29.0% | 25.3% |
| Layer 3→4 | 48.7% | 30.8% |

### 考察

1. **大規模モデルはsaturationが遅い**
   - dim=256では Layer 3→4 で 30.8%（dim=64 では 48.7%）
   - 表現力が高いため、より深い層まで予測が変化し続ける

2. **これは自然な現象**
   - 大規模モデルは複雑なパターンを学習
   - より多くの層で情報を統合
   - 最終的には収束する（そうでなければ訓練が失敗）

3. **Early Exitへの示唆**
   - 大規模モデルでは、より深い層でのexit判定が重要
   - 浅い層でのexit率は低くなる

---

## 実験2: 低ランク近似の精度

**定義**: W を SVD で低ランク分解し、top-1 予測の一致率を測定

### Singular Value Coverage

| Rank | dim=64 | dim=256 |
|------|--------|---------|
| 8 | 85.4% | 77.2% |
| 16 | 87.8% | 78.6% |
| 32 | 92.3% | 80.5% |
| 64 | 100% | 83.8% |
| 128 | - | 89.7% |

### Top-1 Agreement with Full W (Layer 4)

| Rank | dim=64 | dim=256 |
|------|--------|---------|
| 32 | 62.5% | 57.8% |
| 64 | 100% | 70.1% |
| 128 | - | 80.9% |

### 考察

1. **Variance説明率 ≠ Top-1一致率**
   - rank=32 で variance 80-92% をカバーしても、top-1 は 57-62% しか一致しない
   - top-1 は logits の微妙な差で決まるため、低ランク近似に敏感

2. **大規模モデルでは相対的にランクが必要**
   - dim=64 で rank=32（50%）→ 62.5%
   - dim=256 で rank=128（50%）→ 80.9%
   - 絶対ランクではなく相対ランク（rank/dim比）が重要

---

## 実験3: 低ランク Saturation 検出

**目的**: 低ランク W で「saturation（top-1 が変わらない）」を検出できるか

### F1 Score（Layer 3→4）

| Rank | dim=64 | dim=256 |
|------|--------|---------|
| 32 | 65.2% | 55.0% |
| 64 | - | 64.2% |
| 128 | - | **77.0%** |

### 考察

1. **Rank 128（dim=256）で F1=77%** は実用的な水準
2. 低ランク近似でもsaturation検出は可能だが、完璧ではない
3. MLP予測と組み合わせる補助的手段として有望

---

## 実験4: Saturation vs Loss の関係

**仮説**: Saturated トークン（早く確定）は easy（loss が低い）か？

### 結果

| 層 | dim=64 Sat_loss | dim=64 Unsat_loss | dim=256 Sat_loss | dim=256 Unsat_loss |
|----|-----------------|-------------------|------------------|-------------------|
| Layer 1→4 | 6.86 | 6.81 (-0.05) | 7.10 | 7.06 (-0.03) |
| Layer 2→4 | 6.40 | 6.98 (+0.59) | 6.79 | 7.16 (+0.37) |
| Layer 3→4 | 6.50 | 7.11 (+0.62) | 6.40 | 7.37 (+0.96) |

### 考察

1. **Layer 1 は例外**: saturated トークンの方が loss が高い（逆転）
   - Layer 1 での saturation は「確信」ではなく「ノイズ」の可能性
   - Early exit の対象外にすべき

2. **Layer 2 以降は期待通り**: saturated = easy
   - unsaturated トークンは loss が 0.37-0.96 高い
   - saturation 予測は early exit の根拠になる

---

## 実験5: 速度比較

### 結果

| 設定 | Full W | Rank 32 | Rank 64 | Rank 128 |
|------|--------|---------|---------|----------|
| dim=64 | 9.95 ms | 9.64 ms (1.03x) | - | - |
| dim=256 | 10.95 ms | **4.90 ms (2.24x)** | **5.04 ms (2.18x)** | 7.66 ms (1.43x) |

### 考察

1. **小モデル（dim=64）では速度向上なし**
   - 低ランク分解のオーバーヘッドが相殺
   - dim が小さすぎて恩恵がない

2. **大モデル（dim=256）で 2x 以上の速度向上**
   - Rank 32 で 2.24x speedup
   - 実用的な高速化

3. **スケーリング則**
   - dim が大きいほど低ランク近似の恩恵が増加
   - 実際の LLM（dim=4096+）ではさらに効果的と予想

---

## 結論と LEGOへの示唆

### 主要な発見

```
┌────────────────────────────────────────────────────────────────┐
│ 1. 大規模モデルは saturation が遅い（より深い層で確定）        │
│ 2. 低ランク近似は大規模モデルで効果的（2x+ speedup）           │
│ 3. Saturation 検出は F1=77%（rank=128, dim=256）で実用的       │
│ 4. Layer 1 の saturation は信頼できない（exit 対象外）         │
│ 5. Layer 2+ では saturated = easy の関係が成立                 │
└────────────────────────────────────────────────────────────────┘
```

### LEGOフレームワークへの適用

1. **現在の MLP Router（loss 予測）は継続**
   - 30% Oracle を達成済み
   - 安定した動作

2. **Saturation 予測は補助的指標として追加可能**
   - MLP で「top-1 が変わるか」を予測
   - loss 予測と組み合わせてensemble

3. **低ランク W は大規模モデルで有効**
   - dim=256+ で 2x speedup
   - exit 判定の高速化に活用

4. **Layer 1 での early exit は慎重に**
   - saturation 率が低い（10%）
   - loss との相関が逆転

### 今後の実験候補

1. **より大規模なモデル（dim=512, 1024）**でのスケーリング確認
2. **MLP による saturation 予測**（「top-1 が変わるか」を直接予測）
3. **低ランク W + MLP Router の組み合わせ**

---

## 参考文献

- Geva et al., 2022: Saturation Event の発見
- CALM (Schuster et al., 2022): Confident Adaptive Language Modeling
- LayerSkip (Meta, 2024): Early Exit + Self-Speculative Decoding
- EE-LLM (Chen et al., 2023): Large-Scale Early-Exit LLM Framework

---

## 実験スクリプト

```
save_layerwise_data.py          # 小モデル（dim=64）データ生成
save_layerwise_data_large.py    # 大モデル（dim=256）データ生成
analyze_saturation_lowrank.py   # 小モデル分析
analyze_saturation_lowrank_large.py  # 大モデル分析
```

出力ファイル:
- `layerwise_hidden_states.npz`
- `layerwise_hidden_states_large.npz`
- `saturation_lowrank_analysis.png`
- `saturation_lowrank_analysis_large.png`
