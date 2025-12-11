# 論文の方向性メモ

## 現在の主要方針

（既に決まっている方針をここに記載）

---

## 追加案: Deep Supervision ベースのアプローチ

### 概要

Deep Supervision + Inference-time Early Exit の組み合わせ。

- **訓練時**: 全層で損失を計算（Deep Supervision）→ 品質確保
- **推論時**: Confidence-based Early Exit → 計算効率化

### 関連研究

| 研究 | 年 | 概要 |
|------|------|------|
| [DEED](https://arxiv.org/abs/2311.08623) | 2023 | Deep Supervision + Dynamic Early Exit (Amazon) |
| [Depth-Adaptive Transformer](https://arxiv.org/abs/1910.10073) | 2020 | Auxiliary Loss + Early Exit (ICLR) |
| [EE-LLM](https://arxiv.org/abs/2312.04916) | 2024 | Large-scale Early-Exit LLM (ICML) |
| [Early Exit is Natural](https://arxiv.org/abs/2412.01455) | 2024 | Joint optimization無しでもearly exitが機能 |

### 実験結果サマリー

| 閾値 | PPL | Compute | 特徴 |
|------|-----|---------|------|
| 0.8 | 14.04 (-0.2%) | 68.7% | 品質維持で31%削減 |
| 0.0 | 14.51 (+3.1%) | 50.0% | 最大効率 |
| 1.0 | 14.07 | 100.0% | baseline |

詳細: [08_threshold_sweep.md](experiments/08_threshold_sweep.md)

---

## 追加案: HRM (Hierarchical Reasoning Model) の視点

### HRMの特徴: 前回の隠れ状態を利用

HRMは「Iterative Refinement Training」という訓練方式を採用。

```
┌─────────────────────────────────────────────────────────────────┐
│       IRT Transformer (Iterative Refinement Training)           │
├─────────────────────────────────────────────────────────────────┤
│  Segment 1:                                                     │
│  Input → [Embedding] → [Transformer] → [Output] → Loss          │
│                              ↓ state                            │
│  Segment 2:                                                     │
│  Input → [Embedding + state] → [Transformer] → [Output] → Loss  │
│                                                                 │
│  (状態を引き継ぎながら繰り返し、各回で Loss 計算)               │
└─────────────────────────────────────────────────────────────────┘
```

### キーポイント

1. **前回の隠れ状態を次のセグメントに引き継ぐ**
   - 同じ入力を複数回処理
   - 各回で前回の状態を参照して予測を洗練

2. **Iterative Refinement の効果**
   - Standard (12.17) → IRT (11.52) で **5.3% PPL改善**
   - 複数回forward + 各回でloss計算する訓練方式自体が有効

3. **階層構造よりもIRTが重要**
   - IRT Transformer (11.52) vs HRM (11.68)
   - 階層構造の追加効果は限定的

### Deep Supervision vs IRT の比較

| 項目 | Deep Supervision | IRT (Iterative Refinement) |
|------|------------------|----------------------------|
| 損失計算場所 | 各**層**の出力 | 各**セグメント**（時間ステップ）の出力 |
| 補助出力ヘッド | 各中間層に追加 | なし（最終層のみ） |
| 入力 | 各層で異なる（前層の出力）| 全セグメントで同じ入力 x |
| 状態引き継ぎ | なし（層間のみ） | **あり（セグメント間で状態を引き継ぐ）** |
| 目的 | 浅い層も直接学習 | 反復処理による予測の洗練 |

### 統合の可能性

Deep Supervision + IRT の組み合わせ：

```
Segment 1:
  x → [L1] → loss₁ → [L2] → loss₂ → [L3] → loss₃
                                        ↓ state

Segment 2:
  x → [L1 + state] → loss₁' → [L2] → loss₂' → [L3] → loss₃'
```

**仮説**:
- Deep Supervision で各層の表現を強化
- IRT で時間方向の洗練
- 両方の効果を組み合わせられる可能性

---

## 研究の位置づけ整理

### 既存研究との関係

```
                    計算効率化
                        ↑
    MoD (層スキップ)    |    Early Exit (早期終了)
         ○              |         ○
                        |
    ─────────────────── + ───────────────────→ 表現学習
                        |
    Deep Supervision    |    Knowledge Distillation
         ○              |         ○
                        ↓
                    学習効率化


IRT (Iterative Refinement):
  - 時間方向の反復処理
  - 状態の引き継ぎ
  - Deep Supervision と直交する軸
```

### 本プロジェクトの貢献候補

1. **EASE Framework**: Deep Supervision + Early Exit の統一実装
2. **Threshold Analysis**: 推論時閾値の詳細分析
3. **IRT + Deep Supervision**: 未探索の組み合わせ

---

## 今後の検討事項

- [ ] HRMのIRT要素をDeepSupervisionと組み合わせる実験
- [ ] 大規模モデルでの検証
- [ ] 実用的なユースケースの特定
