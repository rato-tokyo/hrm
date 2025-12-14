# Exit Classifier Confidence分布の分析

## 観察結果

Block 0訓練後、validation dataに対するexit classifierのconfidence score分布を可視化した結果、**二峰性（bimodal）分布**が観察された。

### 実験条件
- データセット: WikiText-2
- モデル: dim=64, heads=4, block_layers=(2,)
- 訓練: 8エポック（early stopping）

### 統計情報
| 項目 | 値 |
|------|-----|
| Total tokens | 65,536 |
| Score range | [0.1045, 0.8788] |
| Score mean | 0.5270 |
| Score std | 0.1504 |
| Threshold | 0.5203 |

### 分布の特徴
- 2つの明確な山（peak）が存在
- 左の山: 低confidence（予測困難なトークン）
- 右の山: 高confidence（予測容易なトークン）
- thresholdは2つの山の間に位置

---

## 関連研究

### 1. Bimodal Distribution Removal (BDR)
**論文**: [Bimodal Distribution Removal and Genetic Algorithm in Neural Network](https://arxiv.org/pdf/2002.08729)

この研究が観察と最も一致する：

> 訓練が進むと、予測誤差の分布が二峰性（bimodal）になる。一方の山は「学習済みパターン」、もう一方は「学習困難なパターン（外れ値）」を表す。

| エポック | 分布形状 | 解釈 |
|---------|---------|------|
| Epoch 1 | 単峰性（unimodal） | ランダム重み、何も学習していない |
| Epoch 500 | 二峰性（bimodal） | 左の山=学習済み、右の山=学習困難 |

**LEGOとの対応**:
| BDR研究 | LEGOの観察 |
|---------|-----------|
| 予測誤差の二峰性 | confidence scoreの二峰性 |
| 左の山（低誤差）= 学習済み | 右の山（高confidence）= 予測容易 |
| 右の山（高誤差）= 外れ値 | 左の山（低confidence）= 予測困難 |

### 2. Saturation Event（飽和イベント）
**論文**: [Looking Beyond the Top-1: Transformers Determine Top Tokens in Order](https://arxiv.org/html/2410.20210v1)

Transformerの各層での予測変化を分析した研究：

- **Saturation（飽和）**: top-1予測が固定される層以降でも、モデルは処理を継続
- **順序性**: top-1 → top-2 → top-3...と順番に飽和していく
- **二峰性との関連**: 早期に飽和するトークンと最終層まで飽和しないトークンで分布が分かれる

```
Layer 1: top-1="a"     top-2="the"   ← 順位が不安定
Layer 2: top-1="mat"   top-2="a"     ← まだ変動中
Layer 3: top-1="mat"   top-2="floor" ← top-1が飽和
Layer 4: top-1="mat"   top-2="floor" ← top-2も飽和
```

### 3. Early Exit as a Natural Capability
**論文**: [Early Exit Is a Natural Capability in Transformer-based Models](https://arxiv.org/abs/2412.01455)

- Transformerは本来的にearly exit能力を持つ
- 多くのトークンは最終層に到達する前に十分な信頼度に達する
- joint optimizationや追加レイヤーなしでもearly exitが可能

### 4. Calibration研究
**論文**: [On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599)

- 現代のニューラルネットは過信（overconfident）になりがち
- softmaxが値の差を増幅するため、分布が極端になる傾向
- Temperature Scalingで補正可能

---

## 二峰性分布の解釈

### トークンの二分化

言語モデルにとって、トークンは大きく2種類に分かれる：

| 種類 | 例 | 特徴 | 分布上の位置 |
|------|-----|------|-------------|
| 予測容易 | 句読点、冠詞、頻出パターン | 文脈から容易に推測可能 | 右の山（高confidence） |
| 予測困難 | 固有名詞、専門用語、珍しい表現 | 文脈だけでは困難 | 左の山（低confidence） |

### WikiTextの特性

WikiTextはWikipedia記事ベースのため：
- **定型的な構造**（「〜は〜である」等）→ 予測容易 → 右の山
- **固有名詞・数値・専門用語**が多い → 予測困難 → 左の山

### exit_classifierの学習

exit_classifierはsigmoid出力（0〜1）で、訓練ラベル`exp(-loss)`も0〜1の範囲。MSE損失で訓練すると、ラベル分布が二極化していれば出力も二極化する。

---

## LEGOフレームワークへの示唆

### Hard Example Miningの妥当性

二峰性分布は、LEGOのhard example mining戦略の理論的根拠となる：

1. **右の山（高confidence）のトークン**: Block 0で十分に処理可能 → early exit
2. **左の山（低confidence）のトークン**: より深い層が必要 → Block 1へ

thresholdを2つの山の間に設定することで、自然な分離が可能。

### 層ごとの分布変化（予測）

Saturation Event研究に基づく予測：
- より深い層では右の山（高confidence）が大きくなる
- より多くのトークンが飽和状態に到達する
- 最終層では右の山が支配的になる

### 今後の実験課題

1. **層ごとの分布可視化**: 各層でのconfidence分布を比較
2. **訓練ラベル分布の確認**: `exp(-loss)`自体が二峰性かどうか
3. **データセット依存性**: WikiText以外のデータセットでの分布確認
4. **モデルサイズ依存性**: より大きなモデルでの分布変化

---

## 参考文献

1. [Bimodal Distribution Removal and Genetic Algorithm in Neural Network](https://arxiv.org/pdf/2002.08729) - 二峰性分布と学習パターンの関係
2. [Looking Beyond the Top-1: Transformers Determine Top Tokens in Order](https://arxiv.org/html/2410.20210v1) - Saturation Eventの発見
3. [Early Exit Is a Natural Capability in Transformer-based Models](https://arxiv.org/abs/2412.01455) - Early Exitの本質的能力
4. [On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599) - ニューラルネットの過信問題
