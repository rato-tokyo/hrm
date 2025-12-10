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

## Recommended Architecture

For most use cases: **Infini-HRM** with memory reset per batch
- Best performance (PPL 11.53)
- Moderate parameter increase (+10%)
- Simple integration of long-range memory
