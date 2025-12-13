# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## コア概念

LEGOは、2フェーズ訓練とEarly Exit推論を組み合わせた効率的なTransformer訓練フレームワークです。

### 基本アイデア

1. **Phase 1**: 浅いモデル（2層）を全データで訓練
2. **Hard Example収集**: 信頼度の低いトークンを特定
3. **Phase 2**: 深いモデル（4層）の上位層のみをHard Examplesで訓練
4. **推論**: Early Exitで簡単なトークンは浅い層、難しいトークンは深い層で処理

## 信頼度とルーティング

### 信頼度（Confidence）

- **信頼度** = 予測確率の最大値（max probability）
- 実装: `model.compute_confidence(hidden_state)` → `F.softmax(logits, dim=-1).max(dim=-1).values`

### 閾値の決定

- **自動調整**: 指定比率のトークンがHard Examplesになるよう閾値を計算
- 実装: `compute_confidence_threshold(model, val_batches, target_ratio, device)`
- 例：`hard_example_ratio=0.5` → 信頼度の低い方から50%を「難しいトークン」とする閾値を自動算出

## 2フェーズ訓練の実装

### Phase 1: 浅いモデルの訓練

```python
model = LEGOTransformer(
    vocab_size=vocab_size, dim=dim, num_layers=2, num_heads=num_heads
)
trainer = Trainer(vocab_size=vocab_size, device=device)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)
result = trainer.train_with_early_stopping(model, train_loader, val_loader, optimizer)
```

### Hard Examples収集

```python
# 方法1: 個別関数を使用
confidence_threshold = compute_confidence_threshold(
    model, val_loader, target_ratio=0.5, device=device
)
hard_examples = collect_hard_examples(model, val_loader, confidence_threshold, device)
hard_batches = create_hard_example_loader(hard_examples, batch_size=64)

# 方法2: Trainerのパイプラインメソッドを使用（推奨）
hard_batches, hard_examples, threshold = trainer.collect_hard_examples(
    model, val_loader, target_ratio=0.5, batch_size=64
)
# Returns: (batched_loader, {'inputs', 'hidden_states', 'targets', 'confidences'}, threshold)
```

### Phase 2: 拡張モデルの訓練

```python
# extend メソッドで拡張（重みコピー＋凍結を自動化）
model_extended = model.extend(
    num_layers=4,
    routing_threshold=threshold,
    freeze_lower=True  # Layer 1-2 + embeddingを凍結
).to(device)

# Hard examplesで上位層のみ訓練
optimizer = trainer.create_optimizer(model_extended, base_lr=1e-4)
result = trainer.train_upper_layers_with_early_stopping(
    model_extended, hard_batches, val_loader, hard_examples,
    optimizer, num_lower_layers=2,
    routing_threshold=threshold, exit_layer=2
)
```

### 推論: Two-Stage Routing

```python
# routing_threshold と exit_layer をメソッド引数で指定
stats = trainer.evaluate(
    model_extended, val_loader,
    routing_threshold=threshold, exit_layer=2
)
# Returns: {'acc', 'ppl', 'shallow_ratio', 'compute_cost'}
```

## 公開API

### モデル
- `LEGOTransformer` - 統一モデル（standard/early exitの両モード対応）
  - `forward()` - 標準推論
  - `forward_all_layers()` - 全層出力
  - `forward_train()` - 訓練用（shallow/deep両出力）
  - `forward_inference()` - 推論用（ルーティング付き）
  - `extend()` - 浅いモデルから拡張モデルを作成（インスタンスメソッド）
  - `compute_confidence()` - hidden stateから信頼度を計算

### 訓練
- `Trainer` - 訓練・評価を実行
  - `__init__(vocab_size, device)` - 初期化
  - `evaluate(model, val_batches, routing_threshold=0.0, exit_layer=1)` - 評価
  - `train_with_early_stopping(...)` - Phase 1 訓練
  - `train_upper_layers_with_early_stopping(...)` - Phase 2 訓練
  - `collect_hard_examples(model, val_batches, target_ratio, batch_size)` - Hard Example収集パイプライン

### ユーティリティ関数（削除禁止）
| 関数 | 用途 |
|------|------|
| `set_seed()` | 再現性のためのシード設定 |
| `get_device()` | 利用可能なデバイス取得 |
| `create_synthetic_data()` | テスト用合成データ生成 |
| `compute_confidence_threshold()` | 指定比率で閾値を自動計算 |
| `collect_hard_examples()` | 閾値未満のトークンを収集（トークン単位） |
| `create_hard_example_loader()` | Hard examplesをバッチ化 |
| `train_upper_layers()` | 上位層のみを訓練（hidden statesから直接） |
| `evaluate_on_hard_examples()` | Hard examplesでの評価 |

---

## 開発ルール

### 数値検証の原則

1. **test_lego.pyの数値は厳密一致が必須**
   - テストに記録された期待値は「正解」
   - 1つでも異なればバグとして扱う

2. **リファクタリングの定義**
   - 外部から見た振る舞い（入出力）を変えずに内部構造を改善すること
   - メソッド名・シグネチャ・返り値の構造を変えたらリファクタリングではない

3. **変更時の必須手順**
   - 変更後に `python3 test_lego.py` を実行
   - 12テストすべて合格を確認
   - 数値が異なればコードを修正（テストを変更しない）

### 削除禁止の要素

以下はLEGOの効率性を実現する核心機能であり、削除・簡略化禁止：

1. `collect_hard_examples()` - トークン単位でhidden statesを収集
2. `create_hard_example_loader()` - hidden statesをバッチ化
3. `train_upper_layers()` - hidden statesから直接Layer 3-4を訓練
4. `evaluate_on_hard_examples()` - hard examplesのPPL計算

### コード変更時のチェックリスト

- [ ] `python3 test_lego.py` で12テストすべて合格
- [ ] `python3 -m mypy src/lego/ --ignore-missing-imports` でエラーなし
- [ ] `python3 -m ruff check src/lego/` でエラーなし
- [ ] メソッド名・シグネチャを変更していない
- [ ] 返り値の構造を変更していない

### Git操作

変更完了後は以下を実行：

```bash
git add .
git commit -m "適切なコミットメッセージ"
git push origin main
```
