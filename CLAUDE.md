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
- Layer 2の出力に対するsoftmax後の最大確率値
- 実装: `compute_confidence(model, hidden_state)` → `F.softmax(logits, dim=-1).max(dim=-1).values`

### 閾値の決定

- **自動調整**: 難しいトークンと簡単なトークンが指定比率になるように閾値を計算
- 実装: `compute_confidence_threshold(model, val_batches, target_ratio, device)`
- 例：`hard_example_ratio=0.5` → 信頼度の低い方から50%を「難しいトークン」とする閾値を自動算出
- 閾値計算: `torch.quantile(all_confidences, target_ratio)`

### 簡単なトークン（Easy Tokens）

- Block 1（浅い層）の最終層で閾値以上の信頼度を持つトークン
- 全体の約50%（`hard_example_ratio=0.5`の場合）

### 難しいトークン（Hard Tokens）

- Block 1（浅い層）の最終層で閾値未満の信頼度を持つトークン
- 全体の約50%（`hard_example_ratio=0.5`の場合）
- これらのトークンはより深い処理（Block 2）が必要

## 2つのコアオプション

### 1. StageConfig(layers=(x, y), loss_weight=w)

どの層で損失を計算するかを指定：

- `layers=(2, 2)`: Layer 2で損失計算 → Block 1（Layer 1-2）を訓練
- `layers=(4, 4)`: Layer 4で損失計算 → Block 2（Layer 3-4）を訓練

### 2. routing_threshold + exit_layer

推論時にどのBlockで止めるかを制御：

- `exit_layer`: 早期終了判定を行う層（例：2）
- `routing_threshold`: 信頼度の閾値
- Layer 2の信頼度が閾値以上 → Block 1で終了（簡単なトークン）
- Layer 2の信頼度が閾値未満 → Block 2まで処理（難しいトークン）

## ASHEMConfig

ASHEM (Adaptive Supervision via Hard Example Mining) の設定：

```python
@dataclass
class ASHEMConfig:
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

## ASHEM戦略での使用例

### Phase 1: Block 1訓練（全トークン）

```python
# create_standard_config(num_layers) で生成
config = TrainingConfig(stages=[StageConfig(layers=(2, 2), loss_weight=1.0)])
trainer = Trainer(config, vocab_size=vocab_size, device=device)
result = trainer.train_with_early_stopping(
    model, train_loader, val_loader, optimizer,
    max_epochs=50, patience=1
)
```

- 訓練データ: 全トークン（簡単・難しい両方）
- 損失計算: Layer 2で計算
- 訓練される層: Layer 1-2

### Hard Examples収集（自動閾値調整）

```python
# 閾値の自動計算
confidence_threshold = compute_confidence_threshold(
    model, val_loader, target_ratio=0.5, device=device
)

# 難しいトークンの収集
hard_examples = collect_hard_examples(
    model, val_loader, confidence_threshold, device
)
# Returns: {'inputs', 'hidden_states', 'targets', 'confidences'}
```

- 閾値の自動計算: 信頼度の低い方から50%が「難しいトークン」となる閾値を算出
- 難しいトークンの判定: Layer 2の信頼度 < threshold
- 約50%のトークンを「難しいトークン」として収集

### Phase 2: Block 2訓練（難しいトークンのみ）

```python
# 拡張モデル作成（Early Exit対応）
model_extended = DeepSupervisionTransformer(
    vocab_size=vocab_size, dim=dim, num_layers=4, num_heads=num_heads,
    exit_layer=2, routing_threshold=confidence_threshold
)

# Phase 1の重みをコピー
model_extended.embedding.load_state_dict(model.embedding.state_dict())
for i in range(2):
    model_extended.layers[i].load_state_dict(model.layers[i].state_dict())
model_extended.output_head.load_state_dict(model.output_head.state_dict())

# Hard Freezing: Layer 1-2 + embeddingを凍結
for param in model_extended.embedding.parameters():
    param.requires_grad = False
for i in range(2):
    for param in model_extended.layers[i].parameters():
        param.requires_grad = False

# Phase 2の訓練設定
config = TrainingConfig(stages=[StageConfig(layers=(4, 4), loss_weight=1.0)])

# Hard examplesのデータローダー作成
hard_batches = create_hard_example_loader(hard_examples, batch_size=64)

# 上位層のみ訓練
train_loss = train_upper_layers(
    model_extended, hard_batches, optimizer,
    vocab_size, device, num_lower_layers=2
)
```

- 訓練データ: 難しいトークンのみ（Phase 1で収集した信頼度の低い50%）
- 損失計算: Layer 4で計算
- 訓練される層: Layer 3-4（Layer 1-2 + embeddingは凍結済み）

### 推論: Two-Stage Routing

```python
eval_config = TrainingConfig(
    stages=[StageConfig(layers=(4, 4), loss_weight=1.0)],
    routing_threshold=confidence_threshold,  # Phase 1で計算した閾値を使用
    exit_layer=2
)

eval_trainer = Trainer(eval_config, vocab_size=vocab_size, device=device)
stats = eval_trainer.evaluate(model_extended, val_loader)
# Returns: {'acc', 'ppl', 'shallow_ratio', 'compute_cost'}
```

- 全トークンをLayer 2まで処理
- Layer 2で信頼度を計算
- 簡単なトークン（信頼度 ≥ threshold）→ Layer 2で出力（Block 1で終了）
- 難しいトークン（信頼度 < threshold）→ Layer 4まで処理（Block 2まで継続）

## 重要な独立性

1. **StageConfig**: 訓練時にどの層で損失を計算するか
2. **requires_grad**: どのパラメータを更新するか
3. **routing_threshold + exit_layer**: 推論時にどこで止めるか（自動調整された閾値を使用）

これら3つは完全に独立して制御できます。

## ASHEM関連のユーティリティ関数

| 関数 | 用途 |
|------|------|
| `compute_confidence_threshold()` | 指定比率で閾値を自動計算 |
| `collect_hard_examples()` | 閾値未満のトークンを収集 |
| `create_hard_example_loader()` | Hard examplesをバッチ化 |
| `train_upper_layers()` | 上位層のみを訓練 |
| `evaluate_on_hard_examples()` | Hard examplesでの評価（PPL計算） |

---

## ⚠️ 過去の実装ミスと再発防止

### 2025-12-12 リファクタリングミスの記録

#### 発生した問題

「純粋なLEGO設計」への移行を試みた際に、ASHEM戦略の核心を誤って変更してしまった。

**誤った変更内容:**
1. `collect_hard_examples()` をトークン単位からシーケンス単位に変更
2. `train_upper_layers()`, `evaluate_on_hard_examples()`, `create_hard_example_loader()` を削除
3. 標準のTrainerワークフローでPhase 2を訓練するように変更

**結果:**
- Hard examples収集: 51.2% → 100%（ほぼ全シーケンスに1つはhardトークンがある）
- Phase 2 PPL: 829.78 → 1076.02（悪化）
- Hard PPL改善: 75.8% → 悪化

#### 根本原因

**ASHEM戦略の核心を理解していなかった:**

1. **Hard examplesはトークン単位で収集すべき**
   - シーケンス単位で収集すると、1シーケンスに1つでも難しいトークンがあれば全体が「難しい」扱いになる
   - 結果：ほぼ100%のデータが「難しい」と判定され、Phase 2の意味がなくなる

2. **`train_upper_layers()`は削除してはならない**
   - この関数は**hidden statesから直接Layer 3-4を訓練する**
   - 標準のTrainerはトークン→Embedding→Layer 1-2→Layer 3-4と全層を通す
   - Phase 2では既に計算済みのLayer 2出力（hidden states）から開始することで効率化している

3. **「純粋な設計」と「効率的な実装」は異なる**
   - LEGOの2つのコアオプション（StageConfig, routing_threshold）は訓練/推論の設定
   - ASHEMのユーティリティ関数はこれらを**効率的に実現するための実装**
   - 両者は補完関係にあり、ユーティリティ関数を削除することは設計の「純化」ではない

#### 再発防止のためのチェックリスト

リファクタリング前に以下を確認すること：

- [ ] **既存の実験結果を確認したか？** (`dont_delete.md`等)
- [ ] **変更後の期待される結果を明確にしたか？**
- [ ] **関数削除時：その関数が実現する機能を他でカバーできるか？**
- [ ] **トークン/シーケンス単位の違いを理解しているか？**
- [ ] **効率化のための実装（hidden statesの再利用等）を維持しているか？**

#### ASHEMの不変要素（削除禁止）

以下の要素は削除・簡略化してはならない：

1. **`collect_hard_examples()`** - トークン単位でhidden statesとtargetsを収集
2. **`create_hard_example_loader()`** - hidden statesをバッチ化
3. **`train_upper_layers()`** - hidden statesから直接Layer 3-4を訓練
4. **`evaluate_on_hard_examples()`** - hard examplesのPPL計算

これらは「実装の詳細」ではなく、**ASHEMの効率性を実現する核心機能**である。

---

### 2025-12-13 リファクタリング違反の記録

#### 発生した問題

「DRY原則のためにモデルクラスを統合する」という名目で、**リファクタリングではなく仕様変更**を行ってしまった。

**リファクタリングの定義**: 外部から見た振る舞い（入出力）を変えずに、内部構造を改善すること。

**私が行った誤った変更:**
1. `forward_train()` → `forward_with_routing()` にメソッド名を変更
2. 返り値から `'shallow_ratio'` キーを削除
3. `forward_inference()` メソッドを完全に削除
4. `BaseTransformer` → `StandardTransformer`/`DeepSupervisionTransformer` の継承構造を廃止

**結果:**
- dont_delete.mdに記録された実験結果と異なる数値が出力されるようになった
- Phase 1 Epoch 1: Val PPL=1234.27 → 1208.69（差異発生）
- APIが変更されたため、既存コードとの互換性が破壊された

#### 根本原因

**Claudeがリファクタリングと仕様変更を混同しがち:**

1. **「統合」「簡略化」「DRY」という言葉に引きずられる**
   - これらの目標を達成しようとして、振る舞いの変更を正当化してしまう
   - 「より良い設計」のために既存の動作を変えてしまう

2. **振る舞いの同一性を検証しない**
   - 変更後にテストが通っても、それは「機能的な正しさ」の検証であり「同一性」の検証ではない
   - 数値が完全一致するかどうかを確認していない

3. **メソッド名やシグネチャの変更を軽視する**
   - メソッド名の変更は「リファクタリング」ではなく「API変更」である
   - 返り値の構造を変えることも振る舞いの変更である

#### 再発防止のためのルール

**リファクタリング時の厳格なチェックリスト:**

- [ ] **変更前後で同じ入力に対して同じ出力が得られるか？**
- [ ] **メソッド名・シグネチャは維持されているか？**
- [ ] **返り値の構造（キー名、型）は維持されているか？**
- [ ] **既存のテストが変更なしで通るか？**（テストを変更した時点でリファクタリングではない）
- [ ] **dont_delete.md等の記録された数値と完全一致するか？**

**禁止事項:**

1. ⛔ リファクタリングと称してメソッド名を変更する
2. ⛔ リファクタリングと称して返り値の構造を変更する
3. ⛔ リファクタリングと称してメソッドを削除する
4. ⛔ 「より良い設計」を理由に既存の振る舞いを変更する
5. ⛔ テストコードを変更してリファクタリングを「成功」させる

**正しいリファクタリングの例:**

```python
# OK: 内部実装の変更（外部APIは不変）
def forward_train(self, x):
    # 変更前: 3行で書いていた処理
    # 変更後: ヘルパーメソッドに分割
    h = self._compute_shallow(x)
    return self._build_result(h)

# NG: メソッド名の変更（リファクタリングではない）
def forward_with_routing(self, x):  # ← forward_train から名前変更
    ...
```

#### Claudeへの警告

**Claudeは「改善」「統合」「簡略化」を求められると、振る舞いを変えてしまう傾向がある。**

ユーザーが「リファクタリング」を依頼した場合：
1. まず「振る舞いを変えずに内部構造のみ変更する」ことを確認する
2. 変更後に数値の完全一致を検証する
3. テストコードの変更が必要な場合は、それはリファクタリングではないと認識する
