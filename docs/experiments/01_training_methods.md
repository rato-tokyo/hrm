# Experiment 1: Training Methods Comparison

## Goal

Compare Standard training (final layer loss only) vs LPT (Layer-wise Progressive Training, loss at each layer).

## Configuration

| Parameter | Value |
|-----------|-------|
| Train chars | 100,000 |
| Val chars | 10,000 |
| Dimension | 64 |
| Attention Heads | 4 |
| Layers | 3 |
| Learning Rate | 1e-3 |
| Optimizer | AdamW |
| Gradient Clip | 1.0 |
| Early Stopping | Immediate (patience=0) |

## Results

| Model | Parameters | Best PPL | Accuracy | Epoch |
|-------|------------|----------|----------|-------|
| **LPT (3L)** | 200,832 | **30.54** | **46.4%** | 1 |
| Standard (3L) | 200,832 | 34.86 | 44.3% | 1 |
| Standard (1L) | 69,504 | 35.29 | 46.2% | 1 |

## Key Findings

1. **LPT が Standard より 12.4% 改善** (PPL: 30.54 vs 34.86)
2. 3-layer Standard is slightly worse than 1-layer (34.86 vs 35.29)
3. LPT helps deeper models learn more effectively

## Why LPT Works

```
Standard Training:
  L1 → L2 → L3 → Output → Loss

  Only the final output receives gradient signal.
  Intermediate layers learn "whatever helps the next layer".

LPT Training:
  L1 → Output → Loss₁ ─┐
  L2 → Output → Loss₂ ─┼→ Average → Backprop
  L3 → Output → Loss₃ ─┘

  Each layer learns to produce useful predictions.
  Intermediate representations are more "prediction-ready".
```

## Code

```python
# Standard Training
loss = F.cross_entropy(model(x), target)

# LPT Training
outputs = model.forward_all_layers(x)
losses = [F.cross_entropy(out, target) for out in outputs]
loss = sum(losses) / len(losses)  # Normalized
```
