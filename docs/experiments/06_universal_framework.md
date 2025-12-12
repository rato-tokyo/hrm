# LEGO: Layered Adaptive Supervision Hierarchy

## Overview

**LEGO** (Layered Adaptive Supervision Hierarchy) は、統一的な学習フレームワークです。

2つの基本モデルと2つのコアオプションで構成されています。

**注**: パッケージ名は現在`ease`ですが、将来的に`lash`へ移行予定です。

---

## Base Models

| モデル | 説明 |
|--------|------|
| `StandardTransformer` | 最終層のみで損失計算 |
| `DeepSupervisionTransformer` | 全層で損失計算 + Early Exit対応 |

---

## Core Options

| オプション | 説明 | Reference |
|-----------|------|-----------|
| `layer_weights` | 層ごとの損失重み | - |
| `routing_threshold` | 推論時Early Exit | Teerapittayanon et al., 2016 |

---

## Usage

```python
import sys
sys.path.insert(0, 'src')

from ease import (
    DeepSupervisionTransformer,
    TrainingConfig,
    Trainer,
    create_standard_config,
    create_deep_supervision_config,
)

# モデル作成
model = DeepSupervisionTransformer(vocab_size=1000, dim=64, num_layers=3)

# 設定
config = TrainingConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},  # 層ごとの損失重み
    routing_threshold=0.95,  # Early Exit閾値
)

# 訓練
trainer = Trainer(config, vocab_size=1000)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)
loss = trainer.train_epoch(model, train_batches, optimizer)

# 評価
stats = trainer.evaluate(model, val_batches)
```

---

## Helper Functions

```python
# Standard LLM設定（最終層のみ）
config = create_standard_config(num_layers=3)
# → layer_weights={1: 0, 2: 0, 3: 1}

# Deep Supervision設定（全層均等）
config = create_deep_supervision_config(num_layers=3)
# → layer_weights={1: 0.33, 2: 0.33, 3: 0.33}
```

---

## References

- Lee, C.-Y., et al. (2015). **Deeply-Supervised Nets**. AISTATS 2015.
- Howard, J. & Ruder, S. (2018). **Universal Language Model Fine-tuning (ULMFiT)**. ACL 2018.
- Teerapittayanon, S., et al. (2016). **BranchyNet**. ICPR 2016.
