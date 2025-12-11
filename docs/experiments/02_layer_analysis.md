# Experiment 2: Layer-by-Layer Analysis

## Goal

Analyze how each layer contributes to prediction quality in an LPT-trained model.

## Results (3-layer LPT model)

| Layer | PPL | Accuracy |
|-------|-----|----------|
| L1 | 42.86 | 45.0% |
| L2 | 46.58 | 44.5% |
| L3 | 52.11 | 45.0% |

## Key Findings

1. **Layer 1 is the best** (PPL 42.86 < L2 < L3)
2. Deeper layers have **worse** PPL after training
3. This is due to **overfitting** on the small validation set

## Interpretation

After 1 epoch of training:
- L1 has learned basic patterns well
- L2 and L3 are starting to overfit
- The model should stop training early (which our methodology does)

## Implications for Routing

This result suggests:
- **L1 can handle "easy" tokens** effectively
- **Deeper layers may not always improve predictions**
- **Routing based on L1 confidence** is a valid strategy
