# Loss予測実験の分析結果

## 実験日: 2024-12-14

## 概要

LEGOフレームワークにおいて、hidden statesからper-token lossを予測し、Easy/Hard tokenを分離する方法を比較検証した。

## 実験設定

```python
ExperimentConfig(
    dim=64,
    num_heads=4,
    ffn_dim=256,
    seq_len=32,
    num_samples=10000,
    block_layers=(2,),  # 2層のみ
)
```

- データセット: WikiText-2
- 訓練: Early stopping (patience=3)
- Best val PPL: 918.86
- 総トークン数: 65,536

## 用語定義

| 用語 | 説明 |
|------|------|
| **Oracle** | 「正解を知っている理想的な判定器」。実際のlossを直接見て分離した場合の結果。これが達成可能な上限。 |
| **Loss Diff** | Hard tokenの平均loss - Easy tokenの平均loss。大きいほど分離が良い。 |
| **Softmax Confidence** | モデル出力のsoftmax確率の最大値（正解トークンの確率）。 |

## BDR（Bimodal Distribution Removal）の再現確認

### 結果

実際のloss分布は二峰性を示し、BDR研究の主張と一致した。

| 指標 | 値 |
|------|-----|
| Loss mean | 6.82 |
| Loss std | 4.48 |
| Loss range | [0.04, 21.26] |

Loss分布のQuantile:
- 25%: 3.02（左の山 = Easy）
- 50%: 6.26
- 75%: 9.62（右の山 = Hard）

### Hard/Easy分離（実際のloss基準）

| 指標 | Easy | Hard |
|------|------|------|
| Loss mean | 3.17 | 10.48 |
| Loss Diff | - | **7.31** |
| Cohen's d | - | **2.82** |

**結論**: BDR研究は再現できた。実際のloss分布は二峰性であり、明確なHard/Easy分離が可能。

## 方法1: exit_classifier（1層Linear）

現在のLEGOフレームワークの実装。hidden statesから1層のLinear層でlossを予測。

### 結果

| 指標 | 値 |
|------|-----|
| Pearson r | 0.174 |
| Loss Diff | 1.25 |
| Oracle比 | 17% |

### 問題点

- 予測が狭い範囲に集中（std=1.14 vs actual std=4.48）
- 二峰性を捉えられていない
- 相関が弱い

## 方法2: MLP（2-3層）

exit_classifierをMLPに置き換えて実験。

### 回帰タスク（lossを直接予測）

| Model | MSE | Corr | R² | Loss Diff |
|-------|-----|------|-----|-----------|
| Linear | 18.93 | 0.258 | 0.065 | - |
| MLP-2 (128) | 17.88 | 0.344 | 0.116 | - |
| MLP-3 (128) | 17.97 | 0.342 | 0.112 | - |
| **MLP-2-wide (256)** | **17.82** | **0.351** | **0.119** | **2.40** |

### 分類タスク（loss <= 3 をEasyとして二値分類）

| Model | Accuracy | F1 | Easy Loss | Hard Loss | Loss Diff |
|-------|----------|-----|-----------|-----------|-----------|
| Linear | 75.5% | 0.857 | 4.20 | 6.95 | 2.75 |
| **MLP-2** | **76.8%** | **0.860** | **4.08** | **7.14** | **3.05** |
| MLP-3 | 76.7% | 0.858 | 4.30 | 7.18 | 2.89 |
| MLP-2-wide | 76.4% | 0.855 | 4.27 | 7.19 | 2.92 |

### 結論

- MLPにしても相関は0.35程度が限界
- 分類問題に変換すると若干改善（Diff: 2.40 → 3.05）
- しかしOracle（7.31）の42%程度しか達成できない

## 方法3: Softmax Confidence（正解トークンの確率）

モデル出力のsoftmax確率（正解トークンの確率）を直接使用。

### 結果

| 指標 | 値 |
|------|-----|
| Correlation (vs loss) | **-0.547** |
| Loss Diff (median threshold) | **7.31** |
| Oracle比 | **100%** |

### Threshold別の結果

| Threshold | Easy% | Hard% | Easy Loss | Hard Loss | Diff |
|-----------|-------|-------|-----------|-----------|------|
| 0.01 | 38.0% | 62.0% | 2.47 | 9.49 | 7.03 |
| 0.02 | 32.5% | 67.5% | 2.16 | 9.06 | 6.91 |
| 0.05 | 24.7% | 75.3% | 1.76 | 8.49 | 6.73 |
| Median | 50.0% | 50.0% | 3.17 | 10.48 | **7.31** |

## 最終比較

| Method | Loss Diff | Oracle比 | 備考 |
|--------|-----------|----------|------|
| exit_classifier (Linear) | 1.25 | 17% | 現在の実装 |
| MLP-2 回帰 | 2.40 | 33% | - |
| MLP-2 分類 | 3.05 | 42% | loss <= 3 threshold |
| **Softmax Confidence** | **7.31** | **100%** | Oracleと同等 |
| Oracle (actual loss) | 7.31 | 100% | 理想的な上限 |

## 結論と提言

### 主要な発見

1. **Softmax ConfidenceがOracleと同等の分離性能を持つ**
   - 相関: r = -0.547（confidence ↑ → loss ↓）
   - Loss Diff: 7.31（Oracleと完全一致）

2. **exit_classifierやMLPは不要**
   - 複雑なモデルを訓練しても、単純なsoftmax confidenceに勝てない
   - hidden statesからlossを予測する問題には根本的な限界がある

3. **理由の考察**
   - Softmax confidence = 正解トークンの予測確率
   - Cross-entropy loss = -log(正解トークンの確率)
   - つまり `loss = -log(confidence)` という直接的な関係がある
   - 予測する必要がなく、計算するだけで良い

### 推奨アクション

**exit_classifierを廃止し、softmax confidenceを使用する**

変更点:
```python
# Before (exit_classifier)
confidence = exit_classifier(hidden_states)  # 訓練が必要

# After (softmax confidence)
probs = F.softmax(logits, dim=-1)
confidence = probs.max(dim=-1).values  # 訓練不要、直接計算
```

利点:
- 追加の訓練不要
- 計算コスト削減（Linear層の順伝播が不要）
- 分離性能が大幅向上（17% → 100%）

### 注意点

- 今回の実験は「正解トークンの確率」を使用（teacher forcing的な設定）
- 推論時は正解が分からないため、「最大確率」を使用する必要がある
- これは既存研究（BERxiT等）でも採用されている標準的な方法

---

## 追加実験: 推論時のConfidence近似（2024-12-15）

### 背景

前回の実験では「正解トークンの確率」を使用したが、推論時は正解がわからない。
そこで「最大確率」や代替指標を検証し、計算コスト削減のための低ランク近似も試した。

### 実験設定

- 同一モデル・データを使用
- GPU: NVIDIA L4（Colab）
- 実行時間: 約3分

### 方法4: 推論時の各種Confidence指標

推論時に使用可能な指標を比較。

| 指標 | 計算式 | Lossとの相関 |
|------|--------|-------------|
| softmax_conf | max(softmax(z)) | -0.247 |
| max_z | max(z) | -0.290 |
| **max_minus_mean** | max(z) - mean(z) | **-0.292** |
| margin | max(z) - second_max(z) | -0.105 |

**結論**: どの指標もlossとの相関は弱い（最大でも0.29）。「正解トークンの確率」(-0.55)には遠く及ばない。

### 方法5: hidden statesからの直接予測

MLPで各指標を予測できるか検証。

| 指標 | Corr(pred, actual) | 予測しやすさ |
|------|-------------------|-------------|
| softmax_conf | 0.827 | 高 |
| max_z | 0.896 | 高 |
| **max_minus_mean** | **0.912** | **最高** |
| margin | 0.791 | 中 |

**発見**: `max_minus_mean`が最も予測しやすい（相関0.91）。

### Easy/Hard分離性能

| Method | Loss Diff | Oracle比 |
|--------|-----------|----------|
| softmax_conf (max prob) | 1.69 | 23% |
| max_z | 1.91 | 26% |
| **max_minus_mean** | **2.00** | **27%** |
| margin | 0.46 | 6% |
| Oracle | 7.31 | 100% |

**結論**: 推論時に使える指標は最大でもOracle比27%。「正解トークンの確率」(100%)との差は埋められない。

### 方法6: 低ランク近似

output_head行列WをSVD分解し、少ない計算でsoftmax confidenceを近似できるか検証。

#### SVD Coverage

| Rank | Coverage | 計算量削減 |
|------|----------|-----------|
| 5 | 60.6% | 10倍 |
| 10 | 64.6% | 6倍 |
| 32 | 81.0% | 2倍 |
| 64 | 100% | 1倍（元と同じ）|

#### softmax confidenceの再現精度

| Rank | Coverage | Corr with exact | 結果 |
|------|----------|-----------------|------|
| 1 | 56.9% | 0.12 | 失敗 |
| 5 | 60.6% | 0.15 | 失敗 |
| 10 | 64.6% | 0.18 | 失敗 |
| 32 | 81.0% | 0.28 | 失敗 |
| 64 | 100% | 1.00 | 完全一致 |

**重要な発見**: Coverage 81%でもsoftmax confidenceの相関は0.28しかない。

#### なぜ低ランク近似が失敗するか

```
Coverageの意味: ||W - W_r||_F / ||W||_F
  → logitsの二乗誤差が小さい

softmax confidenceの問題:
  z_exact  = [10.0, 9.9, 9.8, ...] → max prob ≈ 0.35
  z_approx = [10.0, 10.1, 9.7, ...] → max prob ≈ 0.37 (別トークンが最大に)
```

softmaxのmax操作は「どのトークンが最大か」に敏感。logitsの小さな誤差が、softmax結果を大きく変えてしまう。

**結論**: 低ランク近似はsoftmax confidenceには使えない。

---

## 総合結論

### 訓練時 vs 推論時

| 状況 | 使用可能な情報 | 最良手法 | Oracle比 |
|------|---------------|---------|----------|
| **訓練時** | 正解トークンあり | 正解の確率 | 100% |
| **推論時** | 正解トークンなし | max_minus_mean | 27% |

### 推奨方針

1. **訓練時**: softmax confidence（正解トークンの確率）を使用 → Oracle同等
2. **推論時**: max_minus_mean を使用 → 最良だが限界あり（27%）
3. **低ランク近似**: 使用しない → 効果なし

### 残された課題

推論時のEasy/Hard分離は27%が限界。これは「正解がわからない状況での予測」の本質的な困難さを示している。

改善の方向性:
- より良い特徴量設計
- 複数指標の組み合わせ
- コンテキスト情報の活用

## 参考文献

- [Bimodal Distribution Removal (BDR)](https://arxiv.org/pdf/2002.08729) - 二峰性分布と学習パターン
- [BERxiT: Early Exiting for BERT](https://aclanthology.org/2021.eacl-main.8/) - Confidence-based early exit
- [Early Exit Is a Natural Capability in Transformer-based Models](https://arxiv.org/abs/2412.01455) - Softmax confidence for exit
