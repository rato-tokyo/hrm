# HRM Project - Claude Instructions

## Project Overview

**EASE: Efficient Asymmetric Supervision for Early-Exit Transformers**

完全に柔軟な設定可能フレームワーク。3つのオプションを自由に組み合わせて、あらゆる訓練戦略を実現。

---

## フレームワーク概要

### コアコンセプト: 完全な柔軟性

**L3T (Layer-wise Loss, LR, and Threshold)** の3つのオプションで全てを制御：

| オプション | 説明 | Reference |
|-----------|------|-----------|
| **layer_weights** | 層ごとの損失重み（どの層で学習するか） | - |
| **layer_lr_scales** | 層ごとの学習率スケール | Howard & Ruder, 2018 |
| **routing_threshold** | 推論時Early Exit閾値 | Teerapittayanon et al., 2016 |

**重要**: StandardとDeep Supervisionは単なる設定パターンの違い。同じフレームワークで実現可能。

### 設定例：柔軟な組み合わせ

#### パターン1: Standard Transformer（従来型LLM）
```python
config = TrainingConfig(
    layer_weights={1: 0, 2: 0, 3: 1}  # 最終層のみ
)
```

#### パターン2: Deep Supervision（全層均等）
```python
config = TrainingConfig(
    layer_weights={1: 0.33, 2: 0.33, 3: 0.33}  # 全層で学習
)
```

#### パターン3: Asymmetric + Early Exit（柔軟な設定）
```python
# 6層モデル、Layer 3と6でのみ損失計算 + Early Exit
config = TrainingConfig(
    layer_weights={
        1: 0,    # 学習しない
        2: 0,    # 学習しない
        3: 0.5,  # 学習する（Early Exit候補層）
        4: 0,    # 学習しない
        5: 0,    # 学習しない
        6: 0.5   # 学習する（最終層）
    },
    routing_threshold=0.95,  # 推論時、Layer 3で95%信頼度あれば終了
    exit_layer=3
)
```

#### パターン4: Discriminative Fine-Tuning（層ごとの学習率）
```python
config = TrainingConfig(
    layer_weights={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1},  # 最終層のみ
    layer_lr_scales={1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2, 6: 0.1}  # 深い層ほど低学習率
)
```

#### パターン5: 完全カスタム（全オプション活用）
```python
config = TrainingConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3, 4: 0, 5: 0, 6: 0},  # Layer 1重視
    layer_lr_scales={1: 1.0, 2: 0.5, 3: 0.1, 4: 0.05, 5: 0.02, 6: 0.01},
    routing_threshold=0.9,  # Layer 1で90%信頼度あれば終了
    exit_layer=1
)
```

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

### ヘルパー関数（便利な設定プリセット）

```python
from ease import create_standard_config, create_deep_supervision_config

# Standard LLM設定（最終層のみ）
config = create_standard_config(num_layers=3)
# → layer_weights={1: 0, 2: 0, 3: 1}

# Deep Supervision設定（全層均等）
config = create_deep_supervision_config(num_layers=3)
# → layer_weights={1: 0.33, 2: 0.33, 3: 0.33}
```

**注意**: これらはあくまでプリセット。`TrainingConfig`で自由にカスタマイズ可能。

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

## パフォーマンス最適化

### compute_loss() の自動最適化

**3つのオプションを完全に維持したまま、訓練速度を最適化**:

```python
# 最終層のみ（高速パス使用）
config = TrainingConfig(layer_weights={1: 0, 2: 0, 3: 1})
# → forward() を使用（約8%高速化）

# 複数層（汎用パス使用）
config = TrainingConfig(layer_weights={1: 0.33, 2: 0.33, 3: 0.33})
# → forward_all_layers() を使用

# 非対称（汎用パス使用）
config = TrainingConfig(layer_weights={1: 0.7, 2: 0, 3: 0.3})
# → forward_all_layers() を使用
```

**最適化の仕組み**:
- `layer_weights` を解析し、最終層のみ必要な場合を検出
- 最終層のみの場合 → `forward()` 使用（中間層でoutput_headを実行しない）
- それ以外 → `forward_all_layers()` 使用（従来通り）

**互換性保証**:
- ✅ `layer_weights`: すべてのパターンで動作
- ✅ `layer_lr_scales`: 独立（optimizer側で処理）
- ✅ `routing_threshold`: 独立（評価時のみ使用）

**実測効果**（WikiText-2, 10K samples）:
- 最終層のみ: **8.4%高速化**（25.51秒 → 23.38秒）
- 複数層: 変化なし（すでに最適）

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
