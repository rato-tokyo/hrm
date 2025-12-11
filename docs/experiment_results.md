# HRM Experiment Results

## Overview

This document summarizes the experimental results for HRM (Hierarchical Reasoning Model) variants on language modeling tasks.

## Dataset

- **WikiText-2** (character-level)
- Train: 20,000 characters
- Validation: 5,000 characters
- Vocabulary: 97 ASCII printable characters
- Sequence length: 64

## Experiment 1: HRM vs LCT+OACD

### Model Configurations

| Model | Dimension | Layers | Heads | Parameters |
|-------|-----------|--------|-------|------------|
| HRM | 64 | 1 | 4 | ~152K |
| LCT+OACD (1-Layer) | 64 | 1 | 4 | ~78K |
| LCT+OACD (2-Layer) | 64 | 2 | 4 | ~150K |

### Results

| Model | Parameters | Best Val PPL | Epochs |
|-------|------------|--------------|--------|
| **HRM** | 152K | **11.31** | 7 |
| LCT+OACD (1-Layer) | 78K | 31.40 | 30 |
| LCT+OACD (2-Layer) | 150K | 29.94 | 30 |

### Key Finding: Deep Supervision

HRM uses task loss for ALL layers at EVERY segment. LCT+OACD only uses task loss for the output layer, relying on unsupervised OACD loss for intermediate representations.

---

## Experiment 2: HRM vs Infini-HRM (Long-Range Memory)

### Motivation

Standard HRM can only see the current 64-token window. To extend context beyond this limit, we explored Infini-Attention integration.

### Approach Comparison

Two approaches were tested:

#### Failed Approach: Memory in Hidden States
Adding compressive memory to each HRM layer.

```
Embedding → HRM+Memory → HRM+Memory → Output
            ↑ complex     ↑ complex
```

**Result**: PPL 22-23 (worse than baseline)

#### Successful Approach: Infini-Attention Input Layer
Single Infini-Attention layer at input, standard HRM afterwards.

```
Embedding → Infini-Attention → HRM Layer → HRM Layer → Output
            ↑ memory here      ↑ standard   ↑ standard
```

**Result**: PPL 11.53 (4.7% improvement)

### Model Configurations

| Model | Description | Parameters |
|-------|-------------|------------|
| HRM (standard) | Baseline, no memory | 152K |
| HRM+Memory (reset) | Memory in hidden states, reset per batch | 193K |
| HRM+Memory (carry) | Memory in hidden states, carry across batches | 193K |
| Infini-HRM (reset) | Infini-Attention input layer, reset per batch | 168K |
| Infini-HRM (carry) | Infini-Attention input layer, carry across batches | 168K |

### Results

| Model | Parameters | Best Val PPL | Improvement |
|-------|------------|--------------|-------------|
| HRM (standard) | 152K | 12.10 | baseline |
| HRM+Memory (reset) | 193K | 22.61 | -86.9% |
| HRM+Memory (carry) | 193K | 23.24 | -92.1% |
| **Infini-HRM (reset)** | 168K | **11.53** | **+4.7%** |
| Infini-HRM (carry) | 168K | 12.03 | +0.6% |

### Key Findings

1. **Input-layer memory is effective**: Infini-HRM (reset) achieved 4.7% PPL improvement
2. **Hidden-state memory failed**: Adding memory to HRM layers degraded performance significantly
3. **Memory reset per batch works better**: Carrying memory across shuffled batches hurt performance
4. **Modest parameter increase**: 152K → 168K (+10%) for 4.7% improvement

### Architecture: Infini-HRM

```
┌─────────────────────────────────────────────────────────┐
│                      Infini-HRM                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Input tokens [batch, 64]                               │
│         ↓                                                │
│   ┌─────────────┐                                        │
│   │  Embedding  │                                        │
│   └─────────────┘                                        │
│         ↓                                                │
│   ┌─────────────────────────────────────────────┐       │
│   │         Infini-Attention Layer              │       │
│   │  ┌─────────────────────────────────────┐    │       │
│   │  │ Local Attention (causal, 64 tokens) │    │       │
│   │  └─────────────────────────────────────┘    │       │
│   │              ↓ gate ↓                       │       │
│   │  ┌─────────────────────────────────────┐    │       │
│   │  │ Memory Retrieval (compressed past)  │←───│── M   │
│   │  └─────────────────────────────────────┘    │       │
│   │              ↓                              │       │
│   │  Memory Update: M += K^T × V               │       │
│   └─────────────────────────────────────────────┘       │
│         ↓                                                │
│   ┌─────────────┐  ┌─────────────┐                      │
│   │ HRM Layer 0 │→ │ HRM Layer 1 │  (standard HRM)      │
│   │   (T=1)     │  │   (T=2)     │                      │
│   └─────────────┘  └─────────────┘                      │
│         ↓                                                │
│   ┌─────────────┐                                        │
│   │ Output Head │                                        │
│   └─────────────┘                                        │
│         ↓                                                │
│   Output [batch, 64, vocab_size]                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Reference Baselines

WikiText-2 full dataset benchmarks:
- GPT-2 Small (117M params): ~30 PPL
- LSTM (10M params): ~100 PPL
- Transformer-XL (257M): ~24 PPL
- **HRM (152K params, 20K chars)**: 12.10 PPL
- **Infini-HRM (168K params, 20K chars)**: 11.53 PPL

---

## Conclusions

1. **HRM outperforms LCT+OACD** due to deep supervision at all layers
2. **Infini-HRM provides modest improvement** with input-layer memory
3. **Keep memory simple**: Input-layer only, not in hidden states
4. **Reset memory per batch** during training with shuffled data

---

## Experiment 3: Iterative Refinement Training の効果分離

### 訓練方式の定義: Iterative Refinement Training

本実験で使用する訓練方式を **Iterative Refinement Training（反復洗練訓練）** と命名する。

#### Iterative Refinement Training とは

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Iterative Refinement Training                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Segment 1:                                                          │
│    入力 x → Model → 出力 y₁ → Loss₁ → backward → パラメータ更新     │
│                ↓ state (detach)                                      │
│                                                                      │
│  Segment 2:                                                          │
│    入力 x → Model → 出力 y₂ → Loss₂ → backward → パラメータ更新     │
│    (+ state)                                                         │
│                ↓ state (detach)                                      │
│                                                                      │
│  Segment N:                                                          │
│    入力 x → Model → 出力 yₙ → Lossₙ → backward → パラメータ更新     │
│    (+ state)           ↑                                             │
│                    最終出力として使用                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 特徴

| 項目 | 説明 |
|------|------|
| **入力** | 同じ入力 x を複数回処理 |
| **状態伝播** | 前のセグメントの隠れ状態を次のセグメントに渡す |
| **損失計算** | 各セグメントで個別に損失を計算・バックプロパゲーション |
| **最終出力** | 最後のセグメントの出力 yₙ のみを使用（累計ではない）|
| **パラメータ** | 全セグメントで同じパラメータを共有（再利用）|

#### 元々の Deep Supervision との違い

| | Deep Supervision（元々） | Iterative Refinement Training |
|--|-------------------------|-------------------------------|
| 損失計算場所 | 各**層**の出力 | 各**セグメント**（時間ステップ）の出力 |
| 補助出力ヘッド | 各中間層に追加 | なし（最終層のみ） |
| 入力 | 各層で異なる（前層の出力）| 全セグメントで同じ入力 x |
| 目的 | 浅い層も直接学習 | 反復処理による予測の洗練 |

#### 直感的理解

「同じ問題を複数回考え直す」訓練方式：
- 1回目: 初期状態から推論 → 荒い予測
- 2回目: 1回目の結果を参考に再推論 → より良い予測
- N回目: 最も洗練された予測

---

### 目的

HRM の性能向上が「階層構造」によるものか「Iterative Refinement Training」によるものかを分離して検証する。

### 比較モデル

| モデル | 説明 |
|--------|------|
| Standard Transformer | 通常の Transformer LM（1回の forward、1回の loss）|
| IRT Transformer | 通常の Transformer + Iterative Refinement Training |
| HRM | 階層構造 + Iterative Refinement Training |

### モデル構成

```
Standard Transformer:
  Input → Embedding → Transformer Layer → Output
  Loss: 1回/バッチ

IRT Transformer (Iterative Refinement Training):
  Input → Embedding → Transformer Layer → Output
  Loss: num_segments 回/バッチ（状態を引き継ぎながら繰り返し forward）

HRM:
  Input → Embedding → [Layer L (T=1), Layer H (T=2)] → Output
  Loss: num_segments 回/バッチ（階層間で状態を共有）
```

### 設定

- Sequence length: 64
- Dimension: 64 (Standard, DeepSup), 48 (HRM - パラメータ数を揃えるため)
- Transformer layers: 1
- Attention heads: 4
- Segments: 2
- Train: 20,000 chars, Val: 5,000 chars

### 結果

| モデル | パラメータ | Best PPL | Epoch | ベースライン比 |
|--------|-----------|----------|-------|---------------|
| Standard Transformer | 78K | 12.17 | 11 | baseline |
| **IRT Transformer** | 82K | **11.52** | 6 | **+5.3% 改善** |
| HRM | 89K | 11.68 | 14 | +4.0% 改善 |

### 重要な発見

1. **Iterative Refinement Training 単体で効果がある**
   - Standard (12.17) → IRT (11.52) で **5.3% PPL 改善**
   - 複数回 forward + 各回で loss を計算する訓練方式自体が有効

2. **階層構造（HRM）の追加効果は限定的**
   - IRT Transformer (11.52) vs HRM (11.68)
   - 同程度のパラメータ数では、階層構造による追加改善は確認できず

3. **HRM は収束が遅い**
   - IRT: 6 epoch で最良
   - HRM: 14 epoch で最良
   - 階層構造は学習が複雑になる可能性

4. **パラメータ効率**
   - IRT Transformer が最もパラメータ効率が良い（82K で 11.52 PPL）

### 考察

HRM の性能向上の主な要因は「Iterative Refinement Training」であり、「階層構造」の寄与は小さい可能性がある。ただし以下の点で HRM の階層構造が有効な場面があるかもしれない：

- より長いシーケンスでの長期依存関係のモデリング
- より複雑な推論タスク
- スケールアップ時の効率性

### アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────┐
│              Standard Transformer                            │
├─────────────────────────────────────────────────────────────┤
│  Input → [Embedding] → [Transformer] → [Output]              │
│                           ↓                                  │
│                       Loss (1回)                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│       IRT Transformer (Iterative Refinement Training)        │
├─────────────────────────────────────────────────────────────┤
│  Segment 1:                                                  │
│  Input → [Embedding] → [Transformer] → [Output] → Loss      │
│                              ↓ state                         │
│  Segment 2:                                                  │
│  Input → [Embedding + state] → [Transformer] → [Output] → Loss │
│                                                              │
│  (状態を引き継ぎながら繰り返し、各回で Loss 計算)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              HRM (Hierarchical Reasoning Model)              │
├─────────────────────────────────────────────────────────────┤
│  Segment 1:                                                  │
│  Input → [Embedding] → [Layer L ←→ Layer H] → [Output] → Loss │
│                              ↓ states                        │
│  Segment 2:                                                  │
│  Input → [Embedding] → [Layer L ←→ Layer H] → [Output] → Loss │
│                                                              │
│  (L は毎ステップ更新、H は T ステップごとに更新)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiment 4: 訓練方式の比較（IRT vs LPT）

### 訓練方式の定義

#### Layer-wise Progressive Training (LPT)

本実験で新たに定義する訓練方式 **Layer-wise Progressive Training（層別漸進訓練）**。

```
┌─────────────────────────────────────────────────────────────────────┐
│            Layer-wise Progressive Training (LPT)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  x → [Layer 1] → Out₁ → Loss₁ ─┐                                    │
│          ↓                      │                                    │
│      [Layer 2] → Out₂ → Loss₂ ─┼→ Total Loss → 1回の backward       │
│          ↓                      │                                    │
│      [Layer 3] → Out₃ → Loss₃ ─┤                                    │
│          ↓                      │                                    │
│      [Layer 4] → Out₄ → Loss₄ ─┘                                    │
│                     ↑                                                │
│               最終出力として使用                                     │
│                                                                      │
│  * 全層で同じ出力ヘッドを共有（補助ヘッドなし）                      │
│  * 1回の forward で全層の出力を取得                                  │
│  * 損失を合計して1回の backward                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 特徴

| 項目 | 説明 |
|------|------|
| **forward回数** | 1回（全層を1回通過）|
| **backward回数** | 1回（合計損失で）|
| **補助ヘッド** | なし（共有出力ヘッドを使用）|
| **損失計算** | 各層の出力 → 共有出力ヘッド → 正解との損失 |
| **最終出力** | 最終層の出力のみ使用 |

#### 訓練方式の比較

| | Standard | IRT | LPT |
|--|----------|-----|-----|
| forward回数 | 1回 | N回（セグメント数）| 1回 |
| backward回数 | 1回 | N回 | 1回 |
| 損失計算場所 | 最終層のみ | 各セグメント | 各層 |
| 状態伝播 | なし | セグメント間で伝播 | なし |
| 補助パラメータ | なし | 初期状態 | なし |

### 設定

- Sequence length: 64
- Dimension: 64
- Transformer layers: 4
- Attention heads: 4
- Segments (IRT): 2
- Train: 20,000 chars, Val: 5,000 chars

### 結果

| モデル | パラメータ | Best PPL | Epoch | ベースライン比 |
|--------|-----------|----------|-------|---------------|
| Standard | 275K | 10.90 | 6 | baseline |
| IRT | 279K | 11.84 | 5 | -8.6% 劣化 |
| **LPT (sum)** | 275K | **10.27** | 10 | **+5.8% 改善** |
| LPT (weighted) | 275K | 10.43 | 7 | +4.3% 改善 |

### 各層の損失推移（LPT sum）

訓練中の各層の損失推移を観察すると、興味深いパターンが見られる：

```
Epoch 1:  L1:3.72  L2:3.48  L3:3.42  L4:3.40  (全層が高い損失)
Epoch 5:  L1:2.39  L2:2.19  L3:2.11  L4:2.09  (徐々に低下)
Epoch 10: L1:2.33  L2:1.90  L3:1.68  L4:1.59  (深い層ほど低い損失)
Epoch 15: L1:2.29  L2:1.72  L3:1.40  L4:1.24  (差が拡大)
```

**観察**: 浅い層（L1）は損失が高止まりし、深い層（L4）は急速に損失が低下する。これは深い層がより洗練された表現を学習していることを示す。

### 重要な発見

1. **LPT が最良の結果**
   - Standard (10.90) → LPT sum (10.27) で **5.8% PPL 改善**
   - 補助ヘッドなしでも各層の監視が有効

2. **IRT は4層モデルでは効果なし**
   - 1層モデル（Exp3）では効果があったが、4層では劣化
   - 深いモデルでは IRT の効果が薄まる可能性

3. **Sum vs Weighted**
   - Sum (10.27) > Weighted (10.43)
   - 等しい重みの方が良い結果

4. **追加パラメータ不要**
   - LPT は Standard と同じパラメータ数で改善

### 考察

| 訓練方式 | 効果のある状況 |
|----------|---------------|
| IRT | 浅いモデル（1-2層）、反復的な推論が必要なタスク |
| LPT | 深いモデル（4層以上）、勾配消失が問題になる場合 |

LPT は元々の Deep Supervision の利点（浅い層への直接的な勾配）を、補助ヘッドなしで実現している。

### アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────────────┐
│ Standard Training                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ x → [L1] → [L2] → [L3] → [L4] → Output → Loss                       │
│                                            ↑                         │
│                                    Only final layer                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ IRT (Iterative Refinement Training)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Segment 1: x → [L1→L2→L3→L4] → Output → Loss₁ → backward            │
│                      ↓ state                                         │
│ Segment 2: x → [L1→L2→L3→L4] → Output → Loss₂ → backward            │
│            (+state)                                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ LPT (Layer-wise Progressive Training)                                │
├─────────────────────────────────────────────────────────────────────┤
│ x → [L1] → SharedHead → Loss₁ ─┐                                    │
│       ↓                         │                                    │
│     [L2] → SharedHead → Loss₂ ─┼→ Sum → backward                    │
│       ↓                         │                                    │
│     [L3] → SharedHead → Loss₃ ─┤                                    │
│       ↓                         │                                    │
│     [L4] → SharedHead → Loss₄ ─┘ ← Final output                     │
│                                                                      │
│ * SharedHead = 同じ出力ヘッドを全層で共有                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Recommended Architecture

For deep models (4+ layers): **LPT Transformer** (Layer-wise Progressive Training)
- Best performance (PPL 10.27, +5.8% improvement)
- No additional parameters
- Easy to implement (shared output head)

For long-range context: **Infini-HRM** with memory reset per batch
- Best for tasks requiring memory beyond sequence length
- Moderate parameter increase (+10%)

For shallow models (1-2 layers): **IRT Transformer** (Iterative Refinement Training)
- Effective for iterative reasoning tasks
- Simple architecture
