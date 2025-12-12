# ASHEM Stage Specification - Detailed Technical Documentation

**Version**: 1.0
**Date**: 2025-12-12
**Status**: Production-ready (Verified with commit fc9b140)

---

## 📌 重要な定義: "Stage" とは何か

### Stage の正確な定義

**Stage (ステージ)** = **訓練フェーズの時間的区切り**

- **時間軸上の区切り**: 異なるタイミングで実行される訓練の段階
- **独立した訓練ループ**: 各Stageは独自のデータ、モデル構成、Early Stoppingを持つ
- **累積的なモデル構築**: 前のStageのモデルを次のStageで拡張・改良

### ASHEM における Stage の実装

ASHEM は **2-Stage Training** を採用：

```
Stage 1 (Phase 1) → Stage 2 (Phase 2)
     ↓                    ↓
時刻 t=0～t₁         時刻 t₁～t₂
```

**重要**: "Stage" と "Phase" は本ドキュメントでは同義語として使用します。

---

## 🔍 Stage の詳細仕様

### Stage 1 (Phase 1): Shallow Model Training

#### 目的
全データで浅層モデルを訓練し、Hard Examples を識別する基準モデルを構築

#### モデル構成
- **層数**: 2層 (phase1_layers=2)
- **モデルクラス**: `StandardTransformer` または `DeepSupervisionTransformer`
- **訓練データ**: 全データ (WikiText-2 10K samples)

#### 訓練設定
```python
# TrainingConfig for Stage 1
config = TrainingConfig(
    layer_weights={1: 0, 2: 1}  # 最終層のみで損失計算
)

# Early Stopping設定
patience = 1  # ASHEMConfig.phase1_patience
learning_rate = 1e-3  # ASHEMConfig.phase1_lr
```

#### 訓練ループ
```python
for epoch in range(max_epochs):
    # 全データで訓練
    train_loss = trainer.train_epoch(model, train_loader, optimizer)

    # 検証
    val_stats = trainer.evaluate(model, val_loader)
    val_ppl = val_stats['ppl']

    # Early Stopping判定
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        save_model(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break  # Early Stopping
```

#### Stage 1 の出力
1. **訓練済み浅層モデル**: 2層のTransformer
2. **Confidence Threshold**: Hard Example識別のための閾値
3. **Hard Examples**: 浅層モデルが苦手とするサンプル集合

#### 期待される結果 (WikiText-2, 10K samples)
```
Best Val PPL: 986.43
Best Val Acc: 16.03%
Hard PPL (2層モデルでのHard examples性能): 2763.69
```

---

### Stage 間の処理: Hard Example Mining

Stage 1 と Stage 2 の間に実行される**重要な中間処理**：

#### 1. Confidence Threshold 計算

**目的**: Hard Examples を識別するための閾値を決定

**実装** (Per-token filtering):
```python
def compute_confidence_threshold(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    target_ratio: float,  # 0.5 = 50% of tokens
    device: str
) -> float:
    """
    Per-token quantile calculation.

    各トークンごとに信頼度を計算し、target_ratio分位点を閾値とする。
    """
    all_confidences = []

    for x, _ in val_batches:
        x = x.to(device)

        # Forward through all layers
        h = model.embedding(x)
        for layer in model.layers:
            h = layer(h)

        # Compute per-token confidence
        logits = model.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # (batch, seq_len)

        # ⚠️ CRITICAL: Flatten to per-token
        all_confidences.append(confidence.view(-1))

    # Compute threshold
    all_confidences = torch.cat(all_confidences)
    threshold = torch.quantile(all_confidences, target_ratio).item()

    return threshold
```

**⚠️ 重要**: `.view(-1)` によるPer-token flatteningが必須

**期待される出力** (WikiText-2, 10K samples, target_ratio=0.5):
```
Threshold: 0.1499
Interpretation: 信頼度 < 0.1499 のトークンを Hard とみなす
```

#### 2. Hard Examples 収集

**目的**: 閾値以下のConfidenceを持つトークンを収集

**実装** (Per-token filtering):
```python
def collect_hard_examples(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Per-token filtering to collect hard examples.

    各トークンを個別に評価し、閾値以下のものを収集。
    """
    hard_inputs = []
    hard_hidden_states = []
    hard_targets = []
    hard_confidences = []

    for x, y in val_batches:
        x, y = x.to(device), y.to(device)

        # Forward through all layers
        h = model.embedding(x)
        for layer in model.layers:
            h = layer(h)

        # Compute per-token confidence
        logits = model.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # (batch, seq_len)

        # ⚠️ CRITICAL: Per-token comparison
        mask = confidence < threshold  # (batch, seq_len)

        # Flatten and filter
        x_flat = x.view(-1)
        h_flat = h.view(-1, h.shape[-1])
        y_flat = y.view(-1)
        confidence_flat = confidence.view(-1)
        mask_flat = mask.view(-1)

        # Collect hard examples
        hard_inputs.append(x_flat[mask_flat])
        hard_hidden_states.append(h_flat[mask_flat])
        hard_targets.append(y_flat[mask_flat])
        hard_confidences.append(confidence_flat[mask_flat])

    return {
        'inputs': torch.cat(hard_inputs),
        'hidden_states': torch.cat(hard_hidden_states),
        'targets': torch.cat(hard_targets),
        'confidences': torch.cat(hard_confidences)
    }
```

**⚠️ 重要**: Threshold計算と同じPer-token方式を使用

**期待される出力** (WikiText-2, 10K samples):
```
Collected hard examples: 32,768 tokens
Average confidence: 0.0653
Actual ratio: 51.2% (target: 50%)
```

---

### Stage 2 (Phase 2): Deep Model Training on Hard Examples

#### 目的
Hard Examples に特化して深層モデルを訓練し、難しいサンプルでの性能向上

#### モデル構成
- **層数**: 4層 (phase2_layers=4)
- **モデルクラス**: `DeepSupervisionTransformer` (Early Exit サポート)
- **訓練データ**: **Hard Examples のみ** (約32,768 tokens)
- **初期化**:
  - Layer 1-2: Stage 1 の重みをコピー (**Frozen**)
  - Layer 3-4: ランダム初期化 (**Trainable**)

#### Hard Freezing 設定
```python
# Freeze lower layers (Stage 1で訓練済み)
for param in model_extended.embedding.parameters():
    param.requires_grad = False

for i in range(phase1_layers):  # i=0,1 (Layer 1-2)
    for param in model_extended.layers[i].parameters():
        param.requires_grad = False

# Layer 3-4 は自動的に trainable (requires_grad=True)
```

**重要**: Hard Freezing = `requires_grad=False` による完全な凍結

#### 訓練設定
```python
# TrainingConfig for Stage 2
phase2_config = TrainingConfig(
    layer_weights={1: 0, 2: 0, 3: 0, 4: 1}  # 最終層のみで損失計算
)

# Early Stopping設定
patience = 3  # ASHEMConfig.phase2_patience
learning_rate = 1e-4  # ASHEMConfig.phase2_lr (Stage 1より低い)
```

#### 訓練ループ
```python
for epoch in range(max_epochs):
    # Hard Examples のみで訓練
    train_loss = train_upper_layers(
        model_extended, hard_batches, optimizer_upper,
        vocab_size, device, num_lower_layers=2
    )

    # ⚠️ CRITICAL: Early Exit を使用して検証
    eval_config = TrainingConfig(
        layer_weights={1: 0, 2: 0, 3: 0, 4: 1},
        routing_threshold=confidence_threshold,  # Stage間で計算した閾値
        exit_layer=2  # Layer 2 で Early Exit 可能
    )
    eval_trainer = Trainer(eval_config, vocab_size, device)
    val_stats = eval_trainer.evaluate(model_extended, val_loader)
    val_ppl = val_stats['ppl']

    # Hard Examples での性能評価
    hard_ppl = evaluate_on_hard_examples(
        model_extended, hard_examples, vocab_size, device,
        batch_size=64, num_lower_layers=2
    )

    print(f"Epoch {epoch+1} - Val PPL: {val_ppl:.2f} | Hard PPL: {hard_ppl:.2f}")

    # Early Stopping判定 (Val PPL基準)
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        save_model(model_extended)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break  # Early Stopping
```

**⚠️ 重要な実装詳細**:
1. **訓練**: Hard Examples のみ使用
2. **検証**: 全データ使用 (Early Exit 有効)
3. **Early Stopping判定**: Val PPL 基準 (**Hard PPL ではない**)

#### Stage 2 の出力
1. **訓練済み深層モデル**: 4層のTransformer (Layer 1-2は凍結、Layer 3-4は訓練済み)
2. **検証性能**: Val PPL, Val Acc
3. **Hard Examples性能**: Hard PPL

#### 期待される結果 (WikiText-2, 10K samples)
```
Best Epoch: 7
Best Val PPL: 829.78 (Early Exit使用時)
Hard PPL: 668.08 (4層モデルでのHard examples性能)
Hard PPL Improvement: +2095.60 (+75.8%)
```

---

## 🔄 Stage間の重要な違い

### データの違い

| Stage | 訓練データ | データ量 | 選択基準 |
|-------|-----------|---------|---------|
| Stage 1 | **全データ** | 100% (~64,000 tokens) | なし |
| Stage 2 | **Hard Examples のみ** | 約50% (~32,768 tokens) | Confidence < Threshold |

### モデルの違い

| Stage | 層数 | 初期化 | 訓練可能パラメータ |
|-------|-----|-------|------------------|
| Stage 1 | 2層 | ランダム初期化 | 100% (全パラメータ) |
| Stage 2 | 4層 | Layer 1-2: コピー<br>Layer 3-4: ランダム | 50% (Layer 3-4のみ) |

### 訓練設定の違い

| Stage | Learning Rate | Patience | 期待されるEpoch数 |
|-------|--------------|----------|------------------|
| Stage 1 | 1e-3 (高い) | 1 (厳しい) | 3-4 epochs |
| Stage 2 | 1e-4 (低い) | 3 (緩い) | 7-10 epochs |

### 評価の違い

| Stage | 評価対象 | Early Exit | 評価指標 |
|-------|---------|-----------|---------|
| Stage 1 | 全データ | なし | Val PPL, Val Acc |
| Stage 2 | 全データ + Hard Examples | **あり** | Val PPL, Val Acc, Hard PPL |

---

## ⚠️ Critical Implementation Details

### 1. Per-token Filtering の必須性

**絶対に守るべきルール**: Threshold計算とFiltering方式を一致させる

#### 正しい実装 (Per-token):
```python
# Threshold計算
confidence = compute_confidence(model, h)  # (batch, seq_len)
all_confidences.append(confidence.view(-1))  # ← Flatten per-token
threshold = torch.quantile(torch.cat(all_confidences), target_ratio)

# Filtering
mask = confidence < threshold  # (batch, seq_len) ← Per-token comparison
```

#### 間違った実装 (混在):
```python
# ❌ Threshold計算: Per-token
confidence = compute_confidence(model, h)  # (batch, seq_len)
all_confidences.append(confidence.view(-1))  # Per-token
threshold = torch.quantile(torch.cat(all_confidences), target_ratio)

# ❌ Filtering: Sequence-level
mask = confidence.mean(dim=1) < threshold  # ← 間違い！方式が異なる
```

**結果**: Hard Examples が正しく収集されず、実験失敗

### 2. Early Exit の必須使用 (Stage 2 評価)

**Stage 2 の検証時は Early Exit を必ず有効化**:

```python
# ✅ 正しい実装
eval_config = TrainingConfig(
    layer_weights={1: 0, 2: 0, 3: 0, 4: 1},
    routing_threshold=confidence_threshold,  # Early Exit有効
    exit_layer=2
)
```

**理由**: Early Exit を使わないと Val PPL が単調減少し、Early Stopping が機能しない

**実験結果との対応**:
- Early Exit なし → Val PPL: 987 → 883 → 845 → ... (単調減少)
- Early Exit あり → Val PPL: 987 → 883 → 845 → 833 → 830 → 830 → **830** (Epoch 7でベスト)

### 3. Early Stopping 判定基準

**Stage 2 では Val PPL を基準にする** (Hard PPL ではない):

```python
# ✅ 正しい実装
if val_ppl < best_val_ppl:
    best_val_ppl = val_ppl
    save_model()
    patience_counter = 0
else:
    patience_counter += 1

# ❌ 間違った実装
if hard_ppl < best_hard_ppl:  # ← Hard PPLは判定基準にしない
    ...
```

**理由**: Val PPL は汎化性能を表す。Hard PPL は過学習しやすい。

---

## 📊 実験結果の検証方法

### dont_delete.md との完全一致確認

以下のメトリクスが **完全に一致** すれば実装は正しい:

#### Stage 1 (Phase 1)
```
✅ Best Acc: 16.03%
✅ Best PPL: 986.43
✅ Best Epoch: 3
✅ Early Stopping: Epoch 4
```

#### Hard Example Mining
```
✅ Confidence Threshold: 0.1499
✅ Collected Hard Examples: 32,768
✅ Average Confidence: 0.0653
✅ Actual Ratio: 51.2%
```

#### Stage 2 (Phase 2) - 訓練経過
```
✅ Epoch 5: Val PPL 829.80 (New best)
✅ Epoch 7: Val PPL 829.78 (New best)
✅ Epoch 8-10: No improvement (1/3, 2/3, 3/3)
✅ Early Stopping: Epoch 10
✅ Best Model: Epoch 7
```

#### Stage 2 (Phase 2) - 最終結果
```
✅ Best Val PPL: 829.78
✅ Hard PPL: 668.08
✅ Hard PPL Improvement: +2095.60 (+75.8%)
```

#### Final Evaluation (Two-Stage Inference)
```
✅ Accuracy: 15.77%
✅ Shallow ratio (Layer 2): 70.4%
✅ Deep ratio (Layer 4): 29.6%
✅ Compute cost: 64.82% of full model
```

### 不一致が発生した場合のデバッグ

#### 問題1: Hard Examples 収集数が異なる
- **原因**: Per-token filtering が正しく実装されていない
- **確認**: `compute_confidence_threshold()` と `collect_hard_examples()` の方式が一致しているか

#### 問題2: Val PPL が単調減少
- **原因**: Early Exit が無効化されている
- **確認**: Stage 2 評価時の `TrainingConfig` に `routing_threshold` と `exit_layer` が設定されているか

#### 問題3: Early Stopping のタイミングが異なる
- **原因**: Early Stopping 判定基準が Val PPL でない
- **確認**: `if val_ppl < best_val_ppl` を使用しているか (Hard PPL ではない)

#### 問題4: Hard PPL の改善率が異なる
- **原因**: Hard Examples の評価方法が間違っている
- **確認**: `evaluate_on_hard_examples()` が正しい `num_lower_layers` を使用しているか

---

## 🎯 Stage の概念的理解

### ASHEM = 2-Stage Training

```
┌─────────────────────────────────────────────────────────┐
│                  ASHEM Training Flow                     │
└─────────────────────────────────────────────────────────┘

Stage 1 (Phase 1)
├─ Input: 全データ (100%)
├─ Model: 2-layer Transformer
├─ Output: 訓練済み浅層モデル + Confidence Threshold
└─ Duration: ~3-4 epochs (Early Stopping: patience=1)

        ↓ (Hard Example Mining)

├─ Compute Confidence Threshold (target_ratio=0.5)
├─ Collect Hard Examples (約50%のトークン)
└─ Identify: 32,768 hard tokens

        ↓

Stage 2 (Phase 2)
├─ Input: Hard Examples のみ (50%)
├─ Model: 4-layer Transformer (Layer 1-2 frozen)
├─ Training: Layer 3-4 のみ訓練
├─ Evaluation: Early Exit 使用 (全データ)
├─ Output: 訓練済み深層モデル
└─ Duration: ~7-10 epochs (Early Stopping: patience=3)

        ↓

Final Inference (Two-Stage Routing)
├─ Easy Examples → Exit at Layer 2 (70.4%)
├─ Hard Examples → Process to Layer 4 (29.6%)
└─ Compute Cost: 64.82% of full model
```

### Staged Deep Supervision (SDS) への拡張 (概念のみ)

**SDS** = N-Stage Training (ASHEMの一般化)

```
Stage 1: 2 layers, all data
   ↓
Stage 2: 4 layers, hard examples (threshold=0.5)
   ↓
Stage 3: 6 layers, very hard examples (threshold=0.2)
   ↓
...
```

**注意**: SDS の実装は未完成。現在は ASHEM (2-Stage) のみ動作確認済み。

---

## 🔧 実装上の推奨事項

### 1. コード構造

```python
# Stage 1
model_stage1 = StandardTransformer(num_layers=2)
result_stage1 = train_stage1(model_stage1, all_data)

# Hard Example Mining
threshold = compute_confidence_threshold(model_stage1, val_data, 0.5)
hard_examples = collect_hard_examples(model_stage1, val_data, threshold)

# Stage 2
model_stage2 = extend_model(model_stage1, num_layers=4)
freeze_lower_layers(model_stage2, num_lower_layers=2)
result_stage2 = train_stage2(model_stage2, hard_examples)

# Final Evaluation
stats = evaluate_two_stage(model_stage2, val_data, threshold)
```

### 2. 設定管理

```python
@dataclass
class ASHEMConfig:
    # Stage 1
    phase1_layers: int = 2
    phase1_lr: float = 1e-3
    phase1_patience: int = 1

    # Hard Example Mining
    hard_example_ratio: float = 0.5

    # Stage 2
    phase2_layers: int = 4
    phase2_lr: float = 1e-4
    phase2_patience: int = 3
```

### 3. 検証とログ

```python
# Stage 1 完了時
print(f"Stage 1 - Best PPL: {phase1_ppl:.2f}")
print(f"Stage 1 - Hard PPL: {phase1_hard_ppl:.2f}")

# Hard Example Mining 完了時
print(f"Threshold: {threshold:.4f}")
print(f"Hard Examples: {len(hard_examples['targets']):,}")

# Stage 2 各エポック
print(f"Epoch {epoch} - Val PPL: {val_ppl:.2f} | Hard PPL: {hard_ppl:.2f}")

# 最終結果
print(f"Hard PPL Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")
```

---

## 📝 用語集

| 用語 | 定義 |
|-----|------|
| **Stage** | 訓練フェーズの時間的区切り。異なるタイミングで実行される訓練の段階。 |
| **Phase** | Stage の同義語 (本ドキュメントでは交換可能) |
| **Hard Example** | モデルの予測信頼度が閾値以下のサンプル (Per-token) |
| **Confidence Threshold** | Hard Example を識別するための信頼度の閾値 |
| **Per-token Filtering** | 各トークンを個別に評価し、閾値比較を行う方式 |
| **Hard Freezing** | `requires_grad=False` による完全なパラメータ凍結 |
| **Early Exit** | 推論時に途中の層で処理を終了する機構 |
| **Two-Stage Routing** | Easy examples は浅層で、Hard examples は深層で処理 |

---

## ✅ チェックリスト

実装時に必ず確認する項目:

### Stage 1
- [ ] モデルは2層
- [ ] 全データで訓練
- [ ] Early Stopping: patience=1
- [ ] Learning Rate: 1e-3

### Hard Example Mining
- [ ] `compute_confidence_threshold()`: Per-token quantile
- [ ] `collect_hard_examples()`: Per-token filtering
- [ ] Threshold計算とFiltering方式が一致
- [ ] 約50%のトークンが収集される

### Stage 2
- [ ] モデルは4層 (Stage 1から拡張)
- [ ] Layer 1-2: 重みコピー + Frozen (`requires_grad=False`)
- [ ] Layer 3-4: ランダム初期化 + Trainable
- [ ] **訓練データ**: Hard Examples のみ
- [ ] **検証データ**: 全データ (Early Exit 使用)
- [ ] Early Stopping: patience=3, Val PPL基準
- [ ] Learning Rate: 1e-4

### Final Evaluation
- [ ] Early Exit 有効
- [ ] Shallow ratio 計算
- [ ] Compute cost 計算

---

## 🚨 絶対に守るべきルール

1. ✅ **Per-token Filtering の一貫性**: Threshold計算とFilteringで同じ方式を使用
2. ✅ **Early Exit の必須使用**: Stage 2 評価時は必ず Early Exit を有効化
3. ✅ **Val PPL 基準の Early Stopping**: Hard PPL ではなく Val PPL で判定
4. ✅ **Hard Freezing の確認**: Layer 1-2 が `requires_grad=False` であることを確認
5. ✅ **データの正確な使用**: Stage 1=全データ、Stage 2=Hard Examples のみ

---

## 📚 参考リンク

- 実装コード: [colab2.py](../colab2.py)
- ASHEM モジュール: [src/ease/ashem.py](../src/ease/ashem.py)
- 実験結果: [dont_delete.md](../dont_delete.md)
- プロジェクト概要: [CLAUDE.md](../CLAUDE.md)

---

**Last Updated**: 2025-12-12
**Verified**: Commit fc9b140 (動作確認済み)
