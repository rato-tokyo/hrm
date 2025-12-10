# HRM Experiment Results

## Overview

This document summarizes the experimental results comparing HRM (Hierarchical Reasoning Model) with LCT+OACD (Lagged Cache Training with Origin-Anchored Centroid Dispersion) on language modeling tasks.

## Dataset

- **WikiText-2** (character-level)
- Train: 20,000 characters
- Validation: 5,000 characters
- Vocabulary: 97 ASCII printable characters
- Sequence length: 64

## Model Configurations

### HRM
- Dimension: 64
- Layers: 1
- Heads: 4
- N (high-level cycles): 1
- T (low-level timesteps): 2
- Segments: 2
- Parameters: ~152K

### LCT+OACD (1-Layer)
- Dimension: 64
- Heads: 4
- Segments: 4
- Parameters: ~78K

### LCT+OACD (2-Layer Hierarchical)
- Dimension: 64
- Heads: 4
- L-layer: updated every step
- H-layer: updated every 2 steps
- Output: concat(L, H)
- Parameters: ~150K

## Results

| Model | Parameters | Best Val PPL | Epochs |
|-------|------------|--------------|--------|
| HRM | 152K | **11.31** | 7 |
| LCT+OACD (1-Layer) | 78K | 31.40 | 30 |
| LCT+OACD (2-Layer) | 150K | 29.94 | 30 |

### Reference Baselines (WikiText-2 full dataset)
- GPT-2 Small (117M params): ~30
- LSTM (10M params): ~100
- Transformer-XL (257M): ~24

## Key Findings

### 1. HRM Significantly Outperforms LCT+OACD
HRM achieves PPL 11.31 vs LCT+OACD's 29.94 (2.6x better) with similar parameter counts.

### 2. Deep Supervision is Critical
The key difference between HRM and LCT+OACD:

| Component | HRM | LCT+OACD |
|-----------|-----|----------|
| Embedding | Task Loss | OACD Loss (unsupervised) |
| Intermediate Layers | Task Loss | OACD Loss (unsupervised) |
| Output Layer | Task Loss | Task Loss |

HRM uses task loss (cross-entropy with target y) for ALL layers at EVERY segment, providing direct supervision throughout the network. LCT+OACD only uses task loss for the output layer, relying on unsupervised OACD loss for intermediate representations.

### 3. Hierarchical Structure Helps Slightly
2-Layer LCT (PPL 29.94) marginally outperformed 1-Layer LCT (PPL 31.40), but the improvement was modest compared to HRM's advantage.

### 4. Causal Masking is Essential
Initial experiments without causal masking showed unrealistically low PPL (~1.13), indicating data leakage where the model could "see" future tokens. Adding proper causal masking fixed this issue.

## Architecture Details

### HRM Forward Pass
```
Input x → Embedding → z_L (initial)
for segment in segments:
    z_L = L_module(x, z_L, z_H)
    z_H = H_module(z_L, z_H)  # every T steps
    y_hat = output(z_L)
    loss = CrossEntropy(y_hat, y)  # Same y for all segments
    loss.backward()
```

### LCT+OACD Forward Pass
```
Phase 1 (OACD Training):
    L, H = random init
    for segment in segments:
        L = L_layer(x, prev_L, H)
        H = H_layer(prev_H, L)  # every 2 steps
        oacd_loss = dispersion_loss - centroid_loss
        update L, H layers

Phase 2 (Task Training):
    hidden_states = collect from Phase 1
    y_hat = output_head(concat(L, H))
    task_loss = CrossEntropy(y_hat, y)
    update output_head only
```

## Conclusion

HRM's superior performance stems from its **Deep Supervision** approach, where task-specific loss guides learning at all layers and all processing steps. This provides much stronger learning signals compared to LCT+OACD's unsupervised intermediate representations.

For language modeling tasks, HRM is the recommended architecture.
