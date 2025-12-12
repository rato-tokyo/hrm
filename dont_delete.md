============================================================
Hard Example Mining + Two-Stage Inference
============================================================
Device: cuda
GPU: NVIDIA L4

Experiment Design:
  Phase 1: Train 2-layer model
  Compute: Auto-adjust threshold to collect 50% hard examples
  Phase 2: Add 2 layers → Train on hard examples
  Eval: Two-stage inference (Layer 2 or Layer 4) using LASH's Early Exit


============================================================
Standard Transformer - Hard Example Mining
============================================================

Phase 1: Train 2-layer model
============================================================
Epoch 1/50 - Train PPL: 3208.8367 | Val PPL: 1155.4391 | Val Acc: 14.53%
  → New best model (val_loss: 7.0522)
Epoch 2/50 - Train PPL: 688.4413 | Val PPL: 995.6865 | Val Acc: 15.84%
  → New best model (val_loss: 6.9034)
Epoch 3/50 - Train PPL: 424.4466 | Val PPL: 986.4297 | Val Acc: 16.03%
  → New best model (val_loss: 6.8941)
Epoch 4/50 - Train PPL: 298.5926 | Val PPL: 1048.5312 | Val Acc: 15.99%
  → No improvement (1/1)

Early stopping triggered at epoch 4
Best model was at epoch 3

Restored best model from epoch 3

Phase 1 Results:
  Best Acc: 16.03%
  Best PPL: 986.43
  Time: 22.41s

============================================================
Computing Confidence Threshold (target ratio: 50%)
============================================================

✓ Computed confidence threshold: 0.1499
  Examples with confidence < 0.1499 will be treated as hard

============================================================
Collecting Hard Examples
============================================================

✓ Collected 32,768 hard examples
  Average confidence: 0.0653
  Actual ratio: 51.2% (target: 50%)

============================================================
Evaluating Phase 1 on Hard Examples
============================================================

✓ Phase 1 Hard PPL: 2763.69
  (vs Overall Val PPL: 986.43)

============================================================
Phase 2: Add 2 layers → Train on hard examples
============================================================

✓ Copied weights from 2-layer model
✓ Layers 3-4 randomly initialized

📊 Hard Freezing Configuration:
  Layer 1-2: Frozen (requires_grad=False)
  Layer 3-4: Trainable

✓ Frozen lower layers
  Trainable params: 4,600,448 / 9,200,896 (50.0%)
  Hard example batches: 512

📊 Training Configuration:
  Learning rate: 1.0e-04
  Patience: 3
  Max epochs: 50
Epoch 1/50 - Train PPL: 2934.9146 | Val PPL: 987.12 | Val Acc: 15.72% | Hard PPL: 1797.55
  → New best (val_ppl: 987.12)
Epoch 2/50 - Train PPL: 1541.0128 | Val PPL: 883.26 | Val Acc: 15.82% | Hard PPL: 1274.81
  → New best (val_ppl: 883.26)
Epoch 3/50 - Train PPL: 1175.0933 | Val PPL: 844.55 | Val Acc: 15.85% | Hard PPL: 1032.28
  → New best (val_ppl: 844.55)
Epoch 4/50 - Train PPL: 987.6181 | Val PPL: 832.64 | Val Acc: 15.78% | Hard PPL: 893.20
  → New best (val_ppl: 832.64)
Epoch 5/50 - Train PPL: 870.6470 | Val PPL: 829.80 | Val Acc: 15.76% | Hard PPL: 798.39
  → New best (val_ppl: 829.80)
Epoch 6/50 - Train PPL: 786.4368 | Val PPL: 829.89 | Val Acc: 15.78% | Hard PPL: 726.42
  → No improvement (1/3)
Epoch 7/50 - Train PPL: 720.2858 | Val PPL: 829.78 | Val Acc: 15.77% | Hard PPL: 668.08
  → New best (val_ppl: 829.78)
Epoch 8/50 - Train PPL: 665.4617 | Val PPL: 830.82 | Val Acc: 15.76% | Hard PPL: 618.80
  → No improvement (1/3)
Epoch 9/50 - Train PPL: 618.4207 | Val PPL: 832.58 | Val Acc: 15.76% | Hard PPL: 575.97
  → No improvement (2/3)
Epoch 10/50 - Train PPL: 577.0879 | Val PPL: 834.16 | Val Acc: 15.79% | Hard PPL: 538.03
  → No improvement (3/3)

Early stopping at epoch 10
Best model was at epoch 7

Restored best model from Phase 2

Phase 2 Results:
  Best Val PPL: 829.78
  Best Hard PPL: 668.08
  Hard PPL Improvement: +2095.60 (+75.8%)
  Time: 66.58s

============================================================
Final Evaluation (Two-Stage Inference)
============================================================

Results:
  Accuracy: 15.77%
  Shallow ratio (Layer 2): 70.4%
  Deep ratio (Layer 4): 29.6%
  Compute cost: 64.82% of full model

============================================================
Comparison
============================================================

Overall Performance:
  Phase 1 (2-layer only):  Acc 16.03% | PPL 986.43
  Two-stage inference:     Acc 15.77% | PPL 829.78
  Accuracy change:         -0.26%
  PPL change:              -156.65

Hard Examples Performance:
  Phase 1 Hard PPL:        2763.69
  Phase 2 Hard PPL:        668.08
  Hard PPL Improvement:    +2095.60 (+75.8%)

Efficiency:
  Shallow ratio (Layer 2): 70.4%
  Deep ratio (Layer 4):    29.6%
  Compute cost:            64.82% of full model

============================================================
Experiment completed!
============================================================
