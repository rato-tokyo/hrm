# Universal Framework の限界

## 概要

Universal Training Framework は多くのモデルを統一的に表現できますが、
いくつかの構造的な制約があります。

**Note**: v2.0 で動的α と Layer-wise Learning Rate を実装しました。

---

## 再現可能なもの

### 1. 層重み付けによる学習方法

```
Loss = Σ weights[i] * L_i_loss
```

以下はすべて `layer_weights` で表現可能:

| 手法 | 表現方法 |
|------|----------|
| Standard LLM | `{1: 0, 2: 0, 3: 1}` |
| LPT | `{1: 1/3, 2: 1/3, 3: 1/3}` |
| Asymmetric | `{1: α, 2: 0, 3: 1-α}` |
| Deep Focus | `{1: 0.3, 2: 0, 3: 0.7}` |
| L2含む | `{1: 0.7, 2: β, 3: 0.3}` |

### 2. 推論時のRouting

```
routing_threshold = 0    → 全トークンがDeep path
routing_threshold = 0.95 → Confidence-based routing
routing_threshold = 1.0  → 全トークンがShallow path
```

### 3. 動的α (NEW in v2.0) ✅

```python
from experiments.universal_trainer import AlphaSchedule

# Linear decay: 0.9 -> 0.5
schedule = AlphaSchedule('linear', start=0.9, end=0.5)

# Cosine annealing
schedule = AlphaSchedule('cosine', start=0.9, end=0.5)

# Step decay
schedule = AlphaSchedule('step', start=0.9, end=0.5, steps=[10, 20])

config = UniversalConfig(
    layer_weights={1: 0.9, 2: 0, 3: 0.1},
    routing_threshold=0.95,
    alpha_schedule=schedule
)
```

### 4. Layer-wise Learning Rate (NEW in v2.0) ✅

```python
config = UniversalConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},
    routing_threshold=0.95,
    layer_lr_scales={1: 1.0, 2: 0.5, 3: 0.1}  # L1 fast, L3 slow
)

# Optimizer creation
trainer = UniversalTrainer(config, vocab_size=vocab_size)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)
# Result: L1=1e-3, L2=5e-4, L3=1e-4
```

**実験結果**:
| Config | PPL | vs Asymmetric (α=0.7) |
|--------|-----|----------------------|
| Decreasing LR (1.0, 0.5, 0.1) | **18.52** | **19.3% 改善** |
| Increasing LR (0.1, 0.5, 1.0) | 21.14 | 7.9% 改善 |
| No Layer-wise LR | 22.95 | baseline |

---

## 再現できないもの

### 1. 学習可能なConfidence Head

**現状**:
```python
confidence = max(softmax(L1_output))
```
固定のsoftmax最大値を使用。

**再現できない例**:
```python
# 学習可能なconfidence予測
confidence = confidence_head(L1_hidden)  # 別のネットワーク
```

**影響**: Confidence推定の精度向上の余地がある

**拡張案**:
```python
class UniversalConfig:
    confidence_method: str = 'max_softmax'  # or 'learned', 'entropy'
```

---

### ~~2. 動的α (学習中にαを変化)~~ → ✅ 実装済み (v2.0)

---

### ~~3. Layer-wise Learning Rate~~ → ✅ 実装済み (v2.0)

---

### 2. Multi-exit Architecture

**現状**:
```python
exit_layer = 1  # 固定の1箇所のみ
```

**再現できない例**:
```
L1 → Exit1 (conf >= 0.99)
  ↓
L2 → Exit2 (conf >= 0.95)
  ↓
L3 → Final Output
```

**影響**: より細かい計算コスト制御が可能

**拡張案**:
```python
class UniversalConfig:
    exit_layers: List[int] = [1]  # 複数可能に
    exit_thresholds: Dict[int, float] = {1: 0.95}
```

---

### 3. Mixture of Experts (MoE)

**現状**:
全トークンが同じ構造（層）を通過。

**再現できない例**:
```
Token 1 → Expert A (FFN variant 1)
Token 2 → Expert B (FFN variant 2)
Token 3 → Expert A
```

**影響**:
- より効率的なパラメータ利用
- トークン種類に特化した処理

**理由**:
アーキテクチャ自体の変更が必要で、重み付けだけでは表現不可能

---

### 4. 知識蒸留 (Knowledge Distillation)

**現状**:
Ground truth labels に対する損失のみ。

**再現できない例**:
```python
# Teacher modelの出力を模倣
teacher_loss = KL_div(student_output, teacher_output)
total_loss = alpha * task_loss + (1-alpha) * teacher_loss
```

**影響**: より効率的な学習が可能

**拡張案**:
```python
class UniversalConfig:
    teacher_model: Optional[nn.Module] = None
    distillation_alpha: float = 0.0
```

---

## 将来の拡張計画

| 優先度 | 機能 | 実装難易度 | 状態 |
|--------|------|-----------|------|
| 高 | Multi-exit | 中 | 未実装 |
| ~~高~~ | ~~動的α~~ | ~~低~~ | ✅ **実装済み v2.0** |
| 中 | 学習可能Confidence | 中 | 未実装 |
| ~~中~~ | ~~Layer-wise LR~~ | ~~低~~ | ✅ **実装済み v2.0** |
| 低 | MoE | 高 | 未実装 |
| 低 | 知識蒸留 | 中 | 未実装 |

---

## 設計哲学

Universal Framework の目標:
1. **シンプルさ**: 最小限のパラメータで多くの手法を表現
2. **拡張性**: 必要に応じて機能追加可能な設計
3. **再現性**: 同じconfigで同じ結果を保証

現状の制限は意図的な設計選択であり、
必要に応じて拡張することで対応可能です。
