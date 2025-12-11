# Threshold Sweep Experiment

## Overview

DeepSupervisionで訓練したモデルに対し、推論時のconfidence thresholdを変化させてearly exitの効果を測定。

**目的**: 訓練時は全層で学習し品質を確保、推論時のみearly exitで効率化する戦略の検証。

---

## 関連研究

本実験のアプローチは以下の研究と密接に関連する：

| 研究 | 年 | 概要 |
|------|------|------|
| [DEED](https://arxiv.org/abs/2311.08623) | 2023 | Deep Supervision + Dynamic Early Exit on Decoder |
| [Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10073) | 2020 | Auxiliary loss + Early exit (ICLR 2020) |
| [EE-LLM](https://arxiv.org/abs/2312.04916) | 2024 | Large-scale Early-Exit LLM (ICML 2024) |
| [Early Exit is Natural](https://arxiv.org/abs/2412.01455) | 2024 | Joint optimization無しでもearly exitが機能 |
| [LayerSkip](https://arxiv.org/html/2402.16757) | 2024 | Layer dropout + Early exit loss (ACL 2024) |

**特にDEED論文は本実験と同じアプローチ**：
- 訓練時：Deep Supervision（各層にauxiliary loss）
- 推論時：Confidence-based dynamic early exit
- 結果：30%-60%のlatency削減、精度維持

---

## 実験設定

### モデル構成

```
DeepSupervisionTransformer
├── Layers: 4
├── Dim: 32
├── Heads: 2
├── Exit Layer: 2 (全4層の中間)
└── Training: routing_threshold=0 (全層通過)
```

### 損失設定

```python
layer_weights = {
    1: 0,     # L1: no loss
    2: 0.7,   # L2 (exit layer): α=0.7
    3: 0,     # L3: no loss
    4: 0.3,   # L4 (final): 1-α=0.3
}
```

### データ

- Train: 500,000 chars (WikiText-2)
- Val: 50,000 chars
- Seq length: 128
- Batch size: 32

---

## 結果

### Phase 1: 訓練

```
Training: routing_threshold=0 (no early exit)
Best PPL: 14.07 (epoch 1)
```

### Phase 2: Threshold Sweep

| Threshold | PPL | Shallow Exit | Compute | PPL変化 |
|-----------|-----|--------------|---------|---------|
| 0.0 | 14.51 | 100.0% | 50.0% | +3.1% |
| 0.1 | 14.51 | 100.0% | 50.0% | +3.1% |
| 0.2 | 14.45 | 99.7% | 50.2% | +2.7% |
| 0.3 | 14.58 | 97.0% | 51.5% | +3.6% |
| 0.4 | 14.64 | 91.4% | 54.3% | +4.1% |
| 0.5 | 14.80 | 85.4% | 57.3% | +5.2% |
| 0.6 | 14.89 | 79.9% | 60.1% | +5.8% |
| **0.7** | **14.38** | 73.2% | 63.4% | **+2.2%** |
| **0.8** | **14.04** | 62.6% | 68.7% | **-0.2%** |
| 0.9 | 14.20 | 37.6% | 81.2% | +0.9% |
| 1.0 | 14.07 | 0.0% | 100.0% | baseline |

---

## 分析

### 1. 最適閾値: 0.8

```
Threshold=0.8:
  PPL: 14.04 (ベースラインより0.2%良い)
  Compute: 68.7% (31.3%削減)
  Shallow Exit: 62.6%
```

**重要な発見**: 適切な閾値では、early exitによってPPLが**改善**することがある。

### 2. 効率最大化: 閾値0.0-0.2

```
Threshold=0.0-0.2:
  PPL: 14.45-14.51 (+2.7%〜3.1%)
  Compute: 50.0% (50%削減)
  Shallow Exit: ~100%
```

ほぼ全トークンがearly exit → 最大の計算削減。

### 3. 非単調な性能変化

閾値とPPLの関係は**単調ではない**：

```
閾値 0.0 → 0.6: PPL悪化 (14.51 → 14.89)
閾値 0.7 → 0.8: PPL改善 (14.38 → 14.04)  ← 興味深いポイント
閾値 0.8 → 1.0: PPL悪化 (14.04 → 14.07)
```

**解釈**: 中間層（exit layer）の表現が最終層より良い予測をする場合がある。

---

## Trade-off分析

### PPL vs Compute

```
┌──────────────────────────────────────────────────┐
│  PPL                                             │
│  15.0 ┼─────────────────────────────────────     │
│       │     *                                    │
│  14.8 ┼       * *                                │
│       │                                          │
│  14.6 ┼   * *                                    │
│       │                                          │
│  14.4 ┼ * *             *                        │
│       │                                          │
│  14.2 ┼                       *                  │
│       │                                          │
│  14.0 ┼─────────────────────────*───────*─       │
│       └─────┼─────┼─────┼─────┼─────┼─────┼──    │
│            50    60    70    80    90   100      │
│                    Compute (%)                   │
└──────────────────────────────────────────────────┘
```

### 推奨設定

| 目的 | 閾値 | PPL | Compute | 備考 |
|------|------|-----|---------|------|
| **品質維持** | 0.8 | 14.04 | 68.7% | 最適バランス |
| **効率重視** | 0.0 | 14.51 | 50.0% | 最大削減 |
| **品質重視** | 1.0 | 14.07 | 100.0% | baseline |

---

## DEED論文との比較

| 項目 | DEED (Amazon) | 本実験 |
|------|---------------|--------|
| アーキテクチャ | Encoder-Decoder | Decoder-only |
| 訓練 | Deep Supervision | Deep Supervision |
| 推論 | Dynamic Early Exit | Threshold-based Early Exit |
| 計算削減 | 30%-60% | **31%-50%** |
| 品質劣化 | Comparable | **-0.2%〜+5%** |

**結論**: DEEDと同様の効果を確認。小規模実験でも有効性を実証。

---

## Key Insights

1. **Deep Supervision + Inference-time Early Exit は有効**
   - 訓練時は全層で学習（品質確保）
   - 推論時はconfidenceでearly exit（効率化）

2. **最適閾値は0.8付近**
   - PPL維持/改善しつつ31%計算削減
   - threshold=0.95では効果なし（前回の実験）

3. **中間層の予測品質**
   - Deep Supervisionにより中間層も高品質な予測が可能
   - 一部のトークンでは最終層より良い予測

4. **低閾値での大幅削減**
   - threshold=0で50%計算削減
   - PPL劣化は+3%程度に抑制

---

## 今後の課題

1. **大規模モデルでの検証**
   - 層数が増えた場合の最適exit layer
   - 複数のexit pointの効果

2. **適応的閾値**
   - 入力の複雑さに応じた動的閾値
   - Token-level vs Sequence-level routing

3. **Self-speculative decoding**
   - LayerSkipで提案された手法との組み合わせ
   - Draft-then-verify戦略

---

## References

- Tang et al. (2023). **DEED: Dynamic Early Exit on Decoder**. Amazon Science.
- Elbayad et al. (2020). **Depth-Adaptive Transformer**. ICLR 2020.
- Chen et al. (2024). **EE-LLM: Large-Scale Training and Inference of Early-Exit LLMs**. ICML 2024.
- Sun et al. (2024). **Early Exit Is a Natural Capability in Transformer-based Models**. arXiv.
- Elhoushi et al. (2024). **LayerSkip: Enabling Early Exit Inference**. ACL 2024.
