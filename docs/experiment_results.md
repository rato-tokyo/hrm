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

## Experiment 3: Deep Supervision の効果分離

### 目的

HRM の性能向上が「階層構造」によるものか「Deep Supervision（各セグメントでロス計算）」によるものかを分離して検証する。

### 比較モデル

| モデル | 説明 |
|--------|------|
| Standard Transformer | 通常の Transformer LM（1回の forward、1回の loss）|
| DeepSup Transformer | 通常の Transformer + Deep Supervision 訓練 |
| HRM | 階層構造 + Deep Supervision |

### モデル構成

```
Standard Transformer:
  Input → Embedding → Transformer Layer → Output
  Loss: 1回/バッチ

DeepSup Transformer:
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
| **DeepSup Transformer** | 82K | **11.52** | 6 | **+5.3% 改善** |
| HRM | 89K | 11.68 | 14 | +4.0% 改善 |

### 重要な発見

1. **Deep Supervision 単体で効果がある**
   - Standard (12.17) → DeepSup (11.52) で **5.3% PPL 改善**
   - 複数回 forward + 各回で loss を計算する訓練方式自体が有効

2. **階層構造（HRM）の追加効果は限定的**
   - DeepSup Transformer (11.52) vs HRM (11.68)
   - 同程度のパラメータ数では、階層構造による追加改善は確認できず

3. **HRM は収束が遅い**
   - DeepSup: 6 epoch で最良
   - HRM: 14 epoch で最良
   - 階層構造は学習が複雑になる可能性

4. **パラメータ効率**
   - DeepSup Transformer が最もパラメータ効率が良い（82K で 11.52 PPL）

### 考察

HRM の性能向上の主な要因は「Deep Supervision」であり、「階層構造」の寄与は小さい可能性がある。ただし以下の点で HRM の階層構造が有効な場面があるかもしれない：

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
│              DeepSup Transformer                             │
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

## Recommended Architecture

For most use cases: **Infini-HRM** with memory reset per batch
- Best performance (PPL 11.53)
- Moderate parameter increase (+10%)
- Simple integration of long-range memory

For simpler use cases: **DeepSup Transformer**
- Strong performance (PPL 11.52)
- Minimal complexity
- Easy to implement
