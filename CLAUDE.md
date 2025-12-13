# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## コア概念

LEGOは、複数のブロック（Block）を組み合わせて効率的なTransformer訓練を実現するフレームワークです。

### Blockとは

- **Block**: 1つ以上の連続した層で構成される独立したstandard LLM
- 各BlockはStandard Transformerと同じ：最終層だけで損失計算
- 例：
  - Block 1 = Layer 1-2 → Layer 2で損失計算
  - Block 2 = Layer 3-4 → Layer 4で損失計算

## トークンの分類

### 信頼度（Confidence）

- **信頼度** = そのトークンの予測確率の最大値（max probability）
- 実装: `compute_confidence(model, hidden_state)` → `F.softmax(logits, dim=-1).max(dim=-1).values`

### 閾値の決定

- **自動調整**: 難しいトークンと簡単なトークンが指定比率になるように閾値を計算
- 実装: `compute_confidence_threshold(model, val_batches, target_ratio, device)`
- 例：`hard_example_ratio=0.5` → 信頼度の低い方から50%を「難しいトークン」とする閾値を自動算出

## 2つのコアオプション

### 1. StageConfig(layers=(x, y), loss_weight=w)

どの層で損失を計算するかを指定：

- `layers=(2, 2)`: Layer 2で損失計算 → Block 1（Layer 1-2）を訓練
- `layers=(4, 4)`: Layer 4で損失計算 → Block 2（Layer 3-4）を訓練

### 2. routing_threshold + exit_layer

推論時にどのBlockで止めるかを制御：

- `exit_layer`: 早期終了判定を行う層（例：2）
- `routing_threshold`: 信頼度の閾値
- 信頼度 ≥ threshold → Block 1で終了（簡単なトークン）
- 信頼度 < threshold → Block 2まで処理（難しいトークン）

## LEGOConfig

```python
@dataclass
class LEGOConfig:
    # Phase 1: Shallow model
    phase1_layers: int = 2
    phase1_lr: float = 1e-3
    phase1_patience: int = 1

    # Hard example collection
    hard_example_ratio: float = 0.5  # Target 50% as hard examples

    # Phase 2: Deep model
    phase2_layers: int = 4  # Total layers
    phase2_lr: float = 1e-4  # Lower LR for fine-tuning
    phase2_patience: int = 3  # Higher patience for new layers
```

## 2フェーズ訓練の実装

### Phase 1: 浅いモデルの訓練

```python
model = LEGOTransformer(
    vocab_size=vocab_size, dim=dim, num_layers=2, num_heads=num_heads
)
config = create_standard_config(num_layers=2)
trainer = Trainer(config, vocab_size=vocab_size, device=device)
result = trainer.train_with_early_stopping(model, train_loader, val_loader, optimizer)
```

### Hard Examples収集

```python
confidence_threshold = compute_confidence_threshold(
    model, val_loader, target_ratio=0.5, device=device
)
hard_examples = collect_hard_examples(model, val_loader, confidence_threshold, device)
# Returns: {'inputs', 'hidden_states', 'targets', 'confidences'}
```

### Phase 2: 拡張モデルの訓練

```python
# extend_from メソッドで拡張（重みコピー＋凍結を自動化）
model_extended = LEGOTransformer.extend_from(
    source_model=model,
    num_layers=4,
    routing_threshold=confidence_threshold,
    freeze_lower=True  # Layer 1-2 + embeddingを凍結
).to(device)

# Hard examplesで上位層のみ訓練
hard_batches = create_hard_example_loader(hard_examples, batch_size=64)
train_loss = train_upper_layers(
    model_extended, hard_batches, optimizer,
    vocab_size, device, num_lower_layers=2
)
```

### 推論: Two-Stage Routing

```python
eval_config = TrainingConfig(
    stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
    routing_threshold=confidence_threshold,
    exit_layer=2
)
eval_trainer = Trainer(eval_config, vocab_size=vocab_size, device=device)
stats = eval_trainer.evaluate(model_extended, val_loader)
# Returns: {'acc', 'ppl', 'shallow_ratio', 'compute_cost'}
```

## 公開API

### モデル
- `LEGOTransformer` - 統一モデル（standard/early exitの両モード対応）
  - `forward()` - 標準推論
  - `forward_all_layers()` - 全層出力（Deep Supervision用）
  - `forward_train()` - 訓練用（shallow/deep両出力）
  - `forward_inference()` - 推論用（ルーティング付き）
  - `extend_from()` - 浅いモデルから拡張モデルを作成

### 訓練設定
- `StageConfig` - 個別ステージの設定
- `TrainingConfig` - 訓練全体の設定
- `Trainer` - 訓練・評価を実行
- `create_standard_config()` - 標準LLM設定を生成
- `create_deep_supervision_config()` - Deep Supervision設定を生成

### ユーティリティ関数（削除禁止）
| 関数 | 用途 |
|------|------|
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
   - 14テストすべて合格を確認
   - 数値が異なればコードを修正（テストを変更しない）

### 削除禁止の要素

以下はLEGOの効率性を実現する核心機能であり、削除・簡略化禁止：

1. `collect_hard_examples()` - トークン単位でhidden statesを収集
2. `create_hard_example_loader()` - hidden statesをバッチ化
3. `train_upper_layers()` - hidden statesから直接Layer 3-4を訓練
4. `evaluate_on_hard_examples()` - hard examplesのPPL計算

### コード変更時のチェックリスト

- [ ] `python3 test_lego.py` で14テストすべて合格
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
