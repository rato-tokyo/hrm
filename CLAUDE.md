# HRM Project - Claude Instructions

## Project Overview

**EASE: Efficient Asymmetric Supervision for Early-Exit Transformers**

Early-Exit Transformer の学習方法を統一的に扱う汎用フレームワーク。

---

## フレームワーク概要

### EASE とは

EASE は Early-Exit Transformer の学習設定を自由に構成できる汎用フレームワークです。

**設定可能なパラメータ**:
- `layer_weights`: 各層への損失重み（任意の配分）
- `routing_threshold`: Early Exit の confidence 閾値
- `exit_layer`: Early Exit を判定する層
- `layer_lr_scales`: 層ごとの学習率スケール
- `alpha_schedule`: 動的な重みスケジュール

**重要**: EASE は特定の設定を推奨しません。最適な設定はタスク、モデル、データセットに依存します。

### 再現可能な既存手法

| 手法 | Reference |
|------|-----------|
| DEED (Deep Supervision + Early Exit) | Tang et al., 2023 |
| Auxiliary Loss Training | Elbayad et al., 2020 |
| Early Exit | Teerapittayanon et al., 2016 |
| Discriminative Fine-Tuning | Howard & Ruder, 2018 |

---

## 用語対応表

| プロジェクト内 (旧) | 学術用語 | Reference |
|-------------------|---------|-----------|
| DeepSupervision | **DEED** | Tang et al., 2023 |
| Standard Routing | **Auxiliary Loss Training** | Elbayad et al., 2020 |
| Confidence-based Routing | **Early Exit** | Teerapittayanon et al., 2016 |
| Layer-wise Learning Rate | **Discriminative Fine-Tuning** | Howard & Ruder, 2018 |
| Dynamic Alpha | **Learning Rate Curriculum** | Croitoru et al., 2024 |

---

## 実験記録

以下は特定の実験設定（3層、WikiText-2、200K chars）での結果です。
**これらは参考値であり、推奨設定ではありません。**

| Model | PPL | 備考 |
|-------|-----|------|
| Discriminative FT | 18.52 | layer_lr_scales={1:1.0, 2:0.5, 3:0.1} |
| Asymmetric (α=0.7) | 22.95 | layer_weights={1:0.7, 2:0, 3:0.3} |
| Auxiliary Loss (α=0.5) | 23.98 | layer_weights={1:0.5, 2:0, 3:0.5} |
| Standard (3L) | 34.86 | baseline |

詳細は `docs/experiments/` を参照。

---

## モデル一覧

| モデル | 訓練方式 | パラメータ | 用途 |
|--------|---------|-----------|------|
| **DEEDTransformer** | 全トークン→両層で損失 | `layer_weights` (α) | 表現学習強化 |
| **TokenRoutedTransformer** | トークンごとに分離損失 | `routing_threshold` | 効率的推論 |
| **MoDTransformer** | top-k選択で層スキップ | `capacity` | 動的計算量 |
| **StandardTransformer** | 最終層のみ損失 | - | ベースライン |

---

## ファイル構成

```
hrm/
├── CLAUDE.md                    # このファイル
├── src/
│   └── ease/                    # EASE フレームワーク
│       ├── __init__.py          # メインエントリポイント
│       ├── models.py            # DEED, TokenRouted, MoD, Standard Transformer
│       ├── trainer.py           # UniversalConfig, UniversalTrainer, AlphaSchedule
│       └── modules/             # コアモジュール
│           ├── norm.py          # RMSNorm
│           ├── attention.py     # MultiHeadAttention, RoPE
│           ├── ffn.py           # GatedLinearUnit
│           └── transformer.py   # TransformerBlock
├── experiments/
│   ├── __init__.py
│   └── utils.py                 # データ準備、シード設定
├── docs/
│   ├── REFERENCES.md            # 学術的参考文献
│   └── experiments/             # 実験結果ドキュメント
└── run_*.py                     # 実験実行スクリプト
```

---

## 使用方法

### DEEDTransformer（α重み付け損失）

```python
import sys
sys.path.insert(0, 'src')

from ease import DEEDTransformer, UniversalConfig, UniversalTrainer

# α重み付け設定
config = UniversalConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},  # α=0.7
    routing_threshold=0.9,
    exit_layer=1,
)

model = DEEDTransformer(vocab_size=1000, dim=64, num_layers=3)
trainer = UniversalTrainer(config, vocab_size=1000)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)

loss, weights = trainer.train_epoch(model, train_batches, optimizer)
stats = trainer.evaluate(model, val_batches)
```

### TokenRoutedTransformer（トークン単位ルーティング）

```python
from ease import TokenRoutedTransformer

# αは不要、thresholdで損失を分離
model = TokenRoutedTransformer(
    vocab_size=1000, dim=64, num_layers=3,
    exit_layer=1, routing_threshold=0.7
)

# 専用の訓練ループを使用（run_token_routing_experiment.py参照）
```

---

## 今後のタスク

- [ ] より大規模なモデルでの検証実験
- [ ] 実際の LLM (Llama 等) での検証
- [ ] MoD (Mixture-of-Depths) との比較実験
