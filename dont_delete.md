From https://github.com/rato-tokyo/hrm
 * branch            main       -> FETCH_HEAD
Already up to date.
============================================================
LEGO: Hard Example Mining + KV Cache Generation
============================================================
Device: cuda

============================================================
LEGOTransformer - Hard Example Mining + KV Cache
============================================================

Phase 1: Train 2-layer model
============================================================
Epoch 1/50 - Train PPL: 3208.8367 | Val PPL: 1155.44 | Val Acc: 14.53%
  → New best (val_ppl: 1155.44)
Epoch 2/50 - Train PPL: 688.4413 | Val PPL: 995.69 | Val Acc: 15.84%
  → New best (val_ppl: 995.69)
Epoch 3/50 - Train PPL: 424.4466 | Val PPL: 986.43 | Val Acc: 16.03%
  → New best (val_ppl: 986.43)
Epoch 4/50 - Train PPL: 298.5926 | Val PPL: 1048.53 | Val Acc: 15.99%
  → No improvement (1/1)

Early stopping at epoch 4
Best model was at epoch 3

Restored best model from epoch 3

Phase 1 Results: Acc 16.03% | PPL 986.43 | Time 22.46s

============================================================
Computing Confidence Threshold (target: 50%)
============================================================

Threshold: 0.1499
Collected 32,768 hard examples (51.2%)
Phase 1 Hard PPL: 2763.69

============================================================
Phase 2: Add 2 layers, train on hard examples
============================================================

Trainable params: 4,600,448 / 9,200,896 (50.0%)
Epoch 1/50 - Train PPL: 2934.9144 | Val PPL: 987.12 | Val Acc: 15.72% | Hard PPL: 1797.55
  → New best (val_ppl: 987.12)
Epoch 2/50 - Train PPL: 1541.0129 | Val PPL: 883.26 | Val Acc: 15.82% | Hard PPL: 1274.81
  → New best (val_ppl: 883.26)
Epoch 3/50 - Train PPL: 1175.0932 | Val PPL: 844.55 | Val Acc: 15.85% | Hard PPL: 1032.28
  → New best (val_ppl: 844.55)
Epoch 4/50 - Train PPL: 987.6182 | Val PPL: 832.64 | Val Acc: 15.78% | Hard PPL: 893.20
  → New best (val_ppl: 832.64)
Epoch 5/50 - Train PPL: 870.6471 | Val PPL: 829.80 | Val Acc: 15.76% | Hard PPL: 798.39
  → New best (val_ppl: 829.80)
Epoch 6/50 - Train PPL: 786.4367 | Val PPL: 829.89 | Val Acc: 15.78% | Hard PPL: 726.42
  → No improvement (1/3)
Epoch 7/50 - Train PPL: 720.2856 | Val PPL: 829.78 | Val Acc: 15.77% | Hard PPL: 668.08
  → New best (val_ppl: 829.78)
Epoch 8/50 - Train PPL: 665.4618 | Val PPL: 830.82 | Val Acc: 15.76% | Hard PPL: 618.80
  → No improvement (1/3)
Epoch 9/50 - Train PPL: 618.4207 | Val PPL: 832.58 | Val Acc: 15.76% | Hard PPL: 575.97
  → No improvement (2/3)
Epoch 10/50 - Train PPL: 577.0879 | Val PPL: 834.16 | Val Acc: 15.79% | Hard PPL: 538.03
  → No improvement (3/3)

Early stopping at epoch 10
Best model was at epoch 7

Restored best model from epoch 7

Phase 2 Results: Hard PPL 668.08 | Time 67.57s
Hard PPL Improvement: +2095.60 (+75.8%)

============================================================
Final Evaluation (Two-Stage Inference)
============================================================

Accuracy: 15.77%
PPL: 829.78
Shallow ratio: 70.4%
Compute cost: 64.82%

============================================================
KV Cache Generation Demo
============================================================

Prompt length: 8 tokens
Generating: 32 new tokens

Generation Time:
  Without cache: 0.2644s
  With cache:    0.1789s
  Speedup:       1.48x

Outputs match: True

============================================================
Summary
============================================================
Phase 1: Acc 16.03% | PPL 986.43
Phase 2: Acc 15.77% | PPL 829.78
Hard PPL: 2763.69 -> 668.08
KV Cache Speedup: 1.48x

============================================================
Experiment completed!
============================================================
