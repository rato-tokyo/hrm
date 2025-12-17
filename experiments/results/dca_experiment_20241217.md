# DCA-LLM 実験結果 (2024-12-17)

## 実験概要

Dual-Context Attention (DCA) をGPT-2ベースのLLMに統合し、ベースラインと比較。

## 実験設定

| 項目 | 値 |
|------|-----|
| データセット | WikiText-2 |
| サンプル数 | 5,000 |
| バッチサイズ | 16 |
| シーケンス長 | 128 |
| 学習率 | 2.5e-4 |
| Early Stopping | patience=1 |
| GPU | NVIDIA L4 (22.2GB) |

### モデル設定

| 項目 | Baseline | DCA-LLM |
|------|----------|---------|
| ベースモデル | GPT-2 | GPT-2 |
| dim | 256 | 256 |
| num_heads | 4 | 4 |
| num_layers | 4 | 4 |
| 追加レイヤー | なし | DCA層×1 |
| パラメータ数 | 16,156,416 | 17,144,065 |

## 結果

### 訓練ログ

#### Baseline (no DCA)
```
Epoch   1/15 | Train PPL:  2220.69 | Val PPL:  1437.04
Epoch   2/15 | Train PPL:  1177.65 | Val PPL:  1293.34
Epoch   3/15 | Train PPL:   935.29 | Val PPL:  1244.47
Epoch   4/15 | Train PPL:   775.67 | Val PPL:  1225.63  ← Best
Epoch   5/15 | Train PPL:   667.15 | Val PPL:  1237.55
Early stopping at epoch 5
```

#### DCA-LLM
```
Epoch   1/15 | Train PPL:  1979.66 | Val PPL:  1409.73
Epoch   2/15 | Train PPL:  1109.12 | Val PPL:  1307.68  ← Best
Epoch   3/15 | Train PPL:   842.74 | Val PPL:  1326.00
Early stopping at epoch 3
```

### 最終結果

| モデル | Train PPL | Val PPL | 訓練時間 |
|--------|-----------|---------|----------|
| Baseline | 736.27 | **1225.63** | 99.3s |
| DCA-LLM | 948.43 | 1307.68 | 61.5s |

**差分**: DCA-LLM は Val PPL で +82.04 (+6.7%) 悪化

## 分析

### DCA-LLMが劣った原因

1. **早すぎるEarly Stopping**
   - DCA-LLM: Epoch 2でベスト → Epoch 3で停止
   - Baseline: Epoch 4でベスト → Epoch 5で停止
   - DCA層の追加パラメータ（約100万）が2エポックでは学習不足

2. **DCA層の効果が不明確**
   - 訓練時はL1（過去コンテキスト圧縮）を使用していない
   - L0のみではベースGPT-2のattentionと冗長
   - 単純な追加レイヤーとして機能

3. **パラメータ効率の悪さ**
   - +6%のパラメータ増加
   - -6.7%の性能低下
   - パラメータあたりの効率が非常に悪い

### 公平性の問題

現在の比較は不公平:
- DCA-LLMは追加レイヤーにより約100万パラメータ多い
- 公平な比較には:
  - Baselineを5層にする
  - または DCA-LLMのベースを3層にする

## 今後の改善案

1. **学習設定の改善**
   - DCA層に別の学習率を適用
   - Early Stopping の patience を増やす（ただしCLAUDE.md方針に反する）

2. **DCAの本来の効果を検証**
   - 長いシーケンス（512+）でL1圧縮の効果を検証
   - ストリーミング推論でのメモリ効率を検証

3. **アーキテクチャの改善**
   - DCA層をGPT-2の内部に統合（後付けではなく）
   - ベースモデルのattentionをDCAで置換

## 結論

現在の実装では、DCAを後付けで追加しても性能向上は見られなかった。
DCAの本来の価値は「長系列のメモリ効率化」にあり、128トークンの短いシーケンスでは効果が発揮されない可能性がある。

次のステップとして、より長いシーケンス（1024+）での実験を推奨。
