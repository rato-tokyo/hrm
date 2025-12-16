# Hard Token訓練比較実験

## 概要

Hard tokensだけで後段LLMを訓練することの妥当性を検証する実験。

**仮説**: Hard tokensに十分な情報が含まれており、後段LLMはHard tokensだけで効率的に学習できる

## 実験設計

### 条件

| 条件 | 訓練データ | 検証データ |
|------|-----------|-----------|
| A: Hard tokens | Hard tokensのhidden states（N個） | Hard tokens (val) |
| B: All tokens | 全トークンの先頭N個のhidden states | Hard tokens (val) ← 同じ |

- 訓練データ数を揃えることで公平な比較を実現
- 検証データは両条件とも同じHard tokens (val)を使用
- All tokensは先頭から順に取得（文脈の連続性を保持）

### 実験パラメータ

| パラメータ | 値 |
|-----------|-----|
| ベースモデル | SmolLM2-135M-Instruct |
| 閾値 | 0.934 |
| 訓練サンプル数 | 1,000 |
| 検証サンプル数 | 200 |
| シーケンス長 | 128 |
| エポック数 | 10 |
| 学習率 | 1e-4 |
| バッチサイズ | 32 |

## 実験結果

### 基本統計

| 項目 | 値 |
|------|-----|
| 訓練データ全トークン数 | 86,784 |
| 訓練データHardトークン数 | 10,082 (11.6%) |
| 検証データ全トークン数 | 17,280 |
| 検証データHardトークン数 | 1,842 (10.7%) |

### 訓練結果

| 条件 | 訓練トークン数 | Val Loss | Val PPL | 訓練時間 |
|------|--------------|----------|---------|---------|
| **Hard tokens** | 10,082 | **7.13** | **1,252** | 36.5秒 |
| All tokens (sampled) | 10,082 | 8.04 | 3,115 | 36.5秒 |

### 比較分析

| 指標 | 値 |
|------|-----|
| PPL差 (Hard - All) | **-1,863** |
| PPL比 (Hard / All) | **0.402** |
| 訓練データ削減率 | 88.4% |

## 結論

### 主要な発見

1. **Hard tokensで訓練した方が2.5倍良い性能**
   - 同じ数のトークンで訓練しても、Hard tokensを選んだ方がHard tokensの予測精度が大幅に向上
   - PPL: 1,252 vs 3,115

2. **Easy tokensは後段LLMの訓練に寄与しない**
   - Easy tokensで訓練しても、Hard tokensの予測には役立たない
   - これは「ノイズ」というよりも「異なるタスク」と解釈できる
   - Easy tokensは前段LLMで処理完了すべきトークン

3. **仮説を強く支持**
   - Hard tokensのhidden statesだけで、後段LLMは効率的に学習できる
   - 88.4%のデータ削減が可能

### CASCADEへの示唆

現在の設計（Hard tokensのみを後段LLMに渡す）は**理にかなっている**：

- Hard tokensは後段LLMにとって最も有用な訓練データ
- Easy tokensを含めると、むしろ性能が下がる
- 計算効率の観点からも、Hard tokensのみを渡すのが最適

### 解釈

```
Easy tokens: 前段LLMが既に「理解」しているトークン
             → 後段LLMに渡しても、新しい情報を学習しない

Hard tokens: 前段LLMが「苦手」なトークン
             → 後段LLMが専門的に学習すべき対象
             → hidden statesに「なぜ難しいか」の情報が含まれている
```

## 実験の再現方法

```bash
python experiments/compare_training_targets.py \
    --num-samples 1000 \
    --val-samples 200 \
    --epochs 10 \
    --threshold 0.934 \
    --lr 1e-4
```

## 関連ドキュメント

- [Hard Token分析から得られた知見](hard_token_analysis_insights.md)

## 実験日

2024年12月16日
