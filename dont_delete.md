From https://github.com/rato-tokyo/hrm
 * branch            main       -> FETCH_HEAD
Already up to date.
============================================================
LEGO: TRUE Early Exit Generation
============================================================
Device: cuda

============================================================
LEGOTransformer - TRUE Early Exit
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

Phase 1 Results: Acc 16.03% | PPL 986.43 | Time 22.41s

============================================================
Computing Confidence Threshold (target: 50%)
============================================================

Threshold: 0.1499
Collected 32,768 hard examples (51.2%)
Phase 1 Hard PPL: 2763.69

============================================================
Phase 2: Add 2 layers, train on hard examples
============================================================

Trainable params: 4,731,776 / 9,332,224 (50.7%)
Epoch 1/50 - Train PPL: 3039.7457 | Val PPL: 1039.04 | Val Acc: 15.59% | Hard PPL: 1732.84
  → New best (val_ppl: 1039.04)
Epoch 2/50 - Train PPL: 1474.4059 | Val PPL: 917.85 | Val Acc: 15.69% | Hard PPL: 1205.18
  → New best (val_ppl: 917.85)
Epoch 3/50 - Train PPL: 1123.0385 | Val PPL: 885.34 | Val Acc: 15.66% | Hard PPL: 985.09
  → New best (val_ppl: 885.34)
Epoch 4/50 - Train PPL: 950.4378 | Val PPL: 876.45 | Val Acc: 15.67% | Hard PPL: 856.11
  → New best (val_ppl: 876.45)
Epoch 5/50 - Train PPL: 838.7987 | Val PPL: 875.91 | Val Acc: 15.71% | Hard PPL: 765.03
  → New best (val_ppl: 875.91)
Epoch 6/50 - Train PPL: 755.5477 | Val PPL: 880.30 | Val Acc: 15.70% | Hard PPL: 694.33
  → No improvement (1/3)
Epoch 7/50 - Train PPL: 688.5745 | Val PPL: 888.40 | Val Acc: 15.70% | Hard PPL: 636.19
  → No improvement (2/3)
Epoch 8/50 - Train PPL: 632.0650 | Val PPL: 897.16 | Val Acc: 15.73% | Hard PPL: 586.51
  → No improvement (3/3)

Early stopping at epoch 8
Best model was at epoch 5

Restored best model from epoch 5

Phase 2 Results: Hard PPL 765.03 | Time 90.31s
Hard PPL Improvement: +1998.65 (+72.3%)

============================================================
Evaluation: Fake Early Exit (both paths computed)
============================================================

Accuracy: 15.71%
PPL: 875.91
Shallow ratio: 70.7%
Compute cost (theoretical): 52.84%

============================================================
TRUE Early Exit Generation Demo
============================================================

Prompt length: 8 tokens
Generating: 32 new tokens
Routing threshold: 0.1499
Exit layer: 2 / 6

Generation Results:
  Time: 0.1567s
  Exit counts: [24, 8]
  Shallow ratio: 75.0%
  ACTUAL compute cost: 50.0%

Comparison with Standard Generation:
  Standard time: 0.1567s
  Early exit time: 0.1567s
  (No speedup - overhead)

============================================================
Summary
============================================================
Phase 1: Acc 16.03% | PPL 986.43
Phase 2: Acc 15.71% | PPL 875.91
Hard PPL: 2763.69 -> 765.03

TRUE Early Exit Stats:
  Shallow ratio: 75.0%
  ACTUAL compute cost: 50.0%
  Compute savings: 50.0%

============================================================
Experiment completed!
============================================================
