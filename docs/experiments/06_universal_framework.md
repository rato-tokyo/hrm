# EASE: Universal Training Framework

## Overview

**EASE** (Efficient Asymmetric Supervision for Early-Exit) は、Early-Exit Transformer の学習を統一的に表現できる汎用フレームワークです。

各層への損失重み付け、Early Exit のルーティング設定、層ごとの学習率を自由に設定できます。
**特定の設定を推奨するものではなく**、様々な学習方法を実験・比較するための基盤を提供します。

**References**: 本フレームワークは以下の既存研究の手法を統合したものです：
- **DEED** (Tang et al., 2023) - Deep Supervision + Dynamic Early Exit
- **Auxiliary Loss Training** (Elbayad et al., 2020)
- **Discriminative Fine-Tuning** (Howard & Ruder, 2018)
- **Early Exit** (Teerapittayanon et al., 2016)

詳細は [REFERENCES.md](../REFERENCES.md) を参照。

---

## Core Concept

```
Loss = Σ weights[i] * L_i_loss

where:
  weights[i] = layer i の重み (0以上の任意の値、自由に設定可能)
  L_i_loss = layer i の出力に対する損失
```

**重要**: 重みの配分（どの層に損失を適用するか、どの程度の重みにするか）は
タスク、モデルサイズ、データセットに依存します。フレームワークは特定の設定を推奨しません。

---

## Configuration

```python
@dataclass
class UniversalConfig:
    layer_weights: Dict[int, float]  # {layer_idx: weight} (1-indexed)
    exit_layer: int = 1              # Early exit layer for routing
    routing_threshold: float = 0.95  # Confidence threshold (0 = no routing)
    layer_lr_scales: Optional[Dict[int, float]] = None  # Layer-wise LR scales
    alpha_schedule: Optional[AlphaSchedule] = None      # Dynamic weight schedule
```

**すべてのパラメータは自由に設定可能**です。最適な値は実験を通じて決定してください。

---

## Usage

```python
import sys
sys.path.insert(0, 'src')

from ease import (
    DEEDTransformer,
    TokenRoutedTransformer,
    UniversalConfig,
    UniversalTrainer,
)

# === DEEDTransformer ===
# 訓練: 全トークンが両方の損失に貢献（α重み付け）
# 推論: confidence-based early exit
config = UniversalConfig(
    layer_weights={1: 0.5, 2: 0.3, 3: 0.2},  # 任意の重み配分 (α)
    routing_threshold=0.9,                    # 推論時の閾値
    exit_layer=1,
)
model = DEEDTransformer(vocab_size=1000, dim=64, num_layers=3)
trainer = UniversalTrainer(config, vocab_size=1000)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)

# Training
loss, weights = trainer.train_epoch(model, train_batches, optimizer)

# Evaluation
stats = trainer.evaluate(model, val_batches)
# Returns: {ppl, acc, shallow_ratio, compute_cost}

# === TokenRoutedTransformer ===
# 訓練: 各トークンは自分のルートの損失のみに貢献（αは不要）
# 推論: confidence-based early exit
model_tr = TokenRoutedTransformer(
    vocab_size=1000, dim=64, num_layers=3,
    exit_layer=1, routing_threshold=0.7
)
# TokenRoutedTransformerはUniversalTrainerを使わず専用の訓練ループを使用
```

---

## 設定可能なパラメータ

### 1. layer_weights: 層ごとの損失重み

各層に任意の重みを設定できます。

```python
# 例: 3層モデル

# 最終層のみ (Standard LLM)
layer_weights = {1: 0, 2: 0, 3: 1}

# 全層均等 (Deep Supervision)
layer_weights = {1: 1/3, 2: 1/3, 3: 1/3}

# 浅い層と深い層のみ
layer_weights = {1: 0.5, 2: 0, 3: 0.5}

# 任意の配分
layer_weights = {1: 0.7, 2: 0.1, 3: 0.2}
```

### 2. routing_threshold: Early Exit の閾値

推論時に confidence がこの閾値以上なら early exit します。

```python
routing_threshold = 0      # Early Exit 無効（全トークンが全層を通過）
routing_threshold = 0.5    # 低閾値（多くのトークンが early exit）
routing_threshold = 0.95   # 高閾値（高 confidence のトークンのみ early exit）
```

### 3. exit_layer: Early Exit の層

どの層で early exit するかを指定します。

```python
exit_layer = 1  # 1層目で exit 判定
exit_layer = 2  # 2層目で exit 判定
```

### 4. layer_lr_scales: 層ごとの学習率スケール

Discriminative Fine-Tuning のための層ごとの学習率設定。

```python
layer_lr_scales = {1: 1.0, 2: 0.5, 3: 0.1}   # 浅い層に高い学習率
layer_lr_scales = {1: 0.1, 2: 0.5, 3: 1.0}   # 深い層に高い学習率
layer_lr_scales = {1: 1.0, 2: 1.0, 3: 1.0}   # 均一（デフォルト）
```

### 5. alpha_schedule: 動的な重みスケジュール

学習中に重みを変化させることができます。

```python
from ease import AlphaSchedule

# 定数
schedule = AlphaSchedule('constant', start=0.7)

# 線形変化
schedule = AlphaSchedule('linear', start=0.9, end=0.5)

# コサインアニーリング
schedule = AlphaSchedule('cosine', start=0.9, end=0.5)
```

---

## 再現可能な既存手法

EASEフレームワークで様々な既存手法を再現できます：

| 手法 | layer_weights (3層例) | routing_threshold | Reference |
|------|----------------------|-------------------|-----------|
| Standard LLM | {1:0, 2:0, 3:1} | 0 | - |
| Deep Supervision | {1:1/3, 2:1/3, 3:1/3} | 0 | Lee et al., 2015 |
| DEED | {1:1/3, 2:1/3, 3:1/3} | 0.7 | Tang et al., 2023 |
| Auxiliary Loss | {1:0.5, 2:0, 3:0.5} | 0.95 | Elbayad et al., 2020 |
| Early Exit | {1:α, 2:0, 3:1-α} | 任意 | Teerapittayanon et al., 2016 |

**注意**: 上記は「再現可能」というだけで、「推奨」ではありません。

---

## Preset Configurations

便宜上、いくつかのプリセットを用意していますが、これらは特定の実験設定に基づくものです。
**プリセットの使用は必須ではなく、カスタム設定を推奨します。**

```python
from ease import PRESETS

# プリセット一覧（参考用）
PRESETS = {
    'standard_llm': ...,
    'deep_supervision': ...,
    'deed': ...,  # Deep Supervision + Early Exit
    'auxiliary_loss': ...,
    'asymmetric': ...,  # 特定実験の結果、推奨値ではない
}
```

---

## Limitations

EASEフレームワークで**再現できない**もの：

1. **学習可能なConfidence Head**: 現在はmax(softmax)固定
2. **Multi-exit**: 複数のexit pointを持つアーキテクチャ
3. **Mixture of Experts**: トークンごとに異なるパスを選択
4. **知識蒸留**: Teacher-Student learning
5. **Mixture-of-Depths**: トークン選択による動的計算（別途MoDTransformerで対応）

詳細は [07_limitations.md](07_limitations.md) を参照。

---

## References

- Tang, Y., et al. (2023). **DEED: Dynamic Early Exit on Decoder**. Amazon Science.
- Elbayad, M., et al. (2020). **Depth-Adaptive Transformer**. ICLR 2020.
- Teerapittayanon, S., et al. (2016). **BranchyNet**. ICPR 2016.
- Howard, J. & Ruder, S. (2018). **Universal Language Model Fine-tuning (ULMFiT)**. ACL 2018.
