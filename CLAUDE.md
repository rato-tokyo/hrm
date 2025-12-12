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

### Early Stopping（訓練時の早期終了）

```python
# Early Stopping付き訓練
result = trainer.train_with_early_stopping(
    model=model,
    train_batches=train_loader,
    val_batches=val_loader,
    optimizer=optimizer,
    max_epochs=100,
    patience=1,  # デフォルト値: 1エポック改善なしで停止
    verbose=True
)
```

**重要ルール**:
- **patienceのデフォルト値は1**
- 検証損失が1エポックでも悪化したら訓練を停止
- 過学習を防ぎ、訓練時間を短縮
- 最良モデルの状態を自動保存・復元

### Perplexity (PPL) の解釈

**正常な値の範囲**:
- **小規模データ（1K サンプル）**: PPL 100-3000 程度
- **中規模データ（10K サンプル）**: PPL 10-1000 程度
- **大規模データ**: PPL 2-100 程度

**計算式**: `PPL = exp(avg_loss)`
- vocab_size=1000のランダム予測: loss ≈ log(1000) ≈ 6.9, PPL ≈ 1000
- loss=7.3 → PPL ≈ 1500（小規模データでは正常）
- loss=2.3 → PPL ≈ 10（十分に学習済み）

**注意**: PPLは指数関数的に増加するため、lossがわずかに高いだけでPPLは大きく見えます。**Accuracyで評価**することを推奨。

---

## References

- Lee et al. (2015) - Deep Supervision
- Howard & Ruder (2018) - Discriminative Fine-Tuning
- Teerapittayanon et al. (2016) - Early Exit

---

## 今後のタスク

- [ ] より大規模なモデルでの検証実験
- [ ] 実際の LLM (Llama 等) での検証
- [ ] フレームワーク名を **L3T** (Layer-wise Loss, LR, and Threshold) に変更
  - 3つのオプション（layer_weights, layer_lr_scales, routing_threshold）を明示的に表現
