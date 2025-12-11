# HRM Project - Claude Instructions

## Project Overview

**EASE: Efficient Asymmetric Supervision for Early-Exit Transformers**

シンプルな学習フレームワーク。2つの基本モデルと3つのオプションで構成。

---

## フレームワーク概要

### 基本モデル（2種類）

| モデル | 説明 |
|--------|------|
| **StandardTransformer** | 最終層のみで損失計算 |
| **DeepSupervisionTransformer** | 全層で損失計算 + Early Exit対応 |

### オプション（3つ）

| オプション | 説明 | Reference |
|-----------|------|-----------|
| `layer_weights` | 層ごとの損失重み | - |
| `layer_lr_scales` | 層ごとの学習率 | Howard & Ruder, 2018 |
| `routing_threshold` | 推論時Early Exit | Teerapittayanon et al., 2016 |

---

## ファイル構成

```
hrm/
├── CLAUDE.md                    # このファイル
├── src/
│   └── ease/                    # EASE フレームワーク
│       ├── __init__.py          # メインエントリポイント
│       ├── models.py            # Standard, DeepSupervision Transformer
│       ├── trainer.py           # TrainingConfig, Trainer
│       └── modules/             # コアモジュール
│           ├── norm.py          # RMSNorm
│           ├── attention.py     # MultiHeadAttention, RoPE
│           ├── ffn.py           # GatedLinearUnit
│           └── transformer.py   # TransformerBlock
├── experiments/
│   ├── __init__.py
│   └── utils.py                 # データ準備、シード設定
├── docs/
│   └── experiments/             # 実験結果ドキュメント
└── run_experiments.py           # 実験実行スクリプト
```

---

## 使用方法

### 基本的な使用例

```python
import sys
sys.path.insert(0, 'src')

from ease import DeepSupervisionTransformer, Trainer, TrainingConfig

# モデル作成
model = DeepSupervisionTransformer(vocab_size=1000, dim=64, num_layers=3)

# 設定
config = TrainingConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},  # 層ごとの損失重み
    layer_lr_scales={1: 1.0, 2: 0.5, 3: 0.1},  # 層ごとの学習率
    routing_threshold=0.95,  # Early Exit閾値
)

# 訓練
trainer = Trainer(config, vocab_size=1000)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)
loss = trainer.train_epoch(model, train_batches, optimizer)

# 評価
stats = trainer.evaluate(model, val_batches)
```

### ヘルパー関数

```python
from ease import create_standard_config, create_deep_supervision_config

# Standard LLM設定
config = create_standard_config(num_layers=3)
# → layer_weights={1: 0, 2: 0, 3: 1}

# Deep Supervision設定
config = create_deep_supervision_config(num_layers=3)
# → layer_weights={1: 0.33, 2: 0.33, 3: 0.33}
```

---

## References

- Lee et al. (2015) - Deep Supervision
- Howard & Ruder (2018) - Discriminative Fine-Tuning
- Teerapittayanon et al. (2016) - Early Exit

---

## 今後のタスク

- [ ] より大規模なモデルでの検証実験
- [ ] 実際の LLM (Llama 等) での検証
