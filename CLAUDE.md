# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## コア概念

LEGOは、**LEGOBlock単位の段階的訓練**と**TRUE Early Exit**推論を組み合わせた効率的なTransformer訓練フレームワークです。

### 基本アイデア（Block-based設計）

```
Block 1 訓練 → Hard Tokens出力 → Block 2 訓練 → Hard Tokens出力 → Block 3 訓練 → ...
```

1. **Block 1**: 最初のブロックを全データで訓練
2. **Hard Token収集**: 訓練後、信頼度の低いトークンを自動出力
3. **Block 2**: `extend()`で追加、Hard Tokensのみで訓練
4. **繰り返し**: 必要に応じてBlock 3, 4...と拡張可能
5. **推論**: TRUE Early Exitで高信頼度トークンは後続Blockを**実際にスキップ**

---

## アーキテクチャ

### LEGOBlock (nn.Module)

```python
class LEGOBlock(nn.Module):
    """複数のTransformerLayerを持つブロック。"""
    def __init__(self, dim: int, num_heads: int, num_layers: int, threshold: float = 1.0):
        self.layers = nn.ModuleList([...])  # TransformerBlockのリスト
        self.threshold = threshold          # Early Exit閾値

    def forward(h) -> h                    # 全レイヤー処理
    def forward_with_cache(h, cache) -> (h, cache)  # KVキャッシュ対応
    def compute_confidence(h) -> (logits, confidence)  # 信頼度計算（トークン単位）
    def freeze() / unfreeze()                    # パラメータ凍結
```

### LEGOTransformer (nn.Module)

```python
class LEGOTransformer(nn.Module):
    """LEGOBlockの管理と橋渡し。"""
    def __init__(self, vocab_size, dim, num_heads, blocks: List[LEGOBlock]):
        self.embedding = nn.Embedding(...)
        self.blocks = nn.ModuleList(blocks)
        self.output_head = nn.Linear(...)

    @classmethod
    def create(vocab_size, dim, num_layers, num_heads, threshold) -> LEGOTransformer
        # 単一Blockモデルを作成

    def forward(x) -> logits                     # 標準推論（全ブロック通過）
    def get_hidden_states(x, up_to_block) -> h   # 指定ブロックまでのhidden states
    def forward_from_block(h, start_block_idx)   # 指定Blockから処理
    def forward_with_routing(x) -> (logits, stats)  # TRUE Early Exit評価
    def generate(...) -> (tokens, stats)         # TRUE Early Exit生成
    def extend(num_new_layers, threshold) -> LEGOTransformer  # Block追加
```

---

## 使用例

### Block 1 訓練

```python
model = LEGOTransformer.create(vocab_size=vocab_size, dim=dim, num_layers=2, num_heads=4)

trainer = Trainer(vocab_size=vocab_size, device=device)
result = trainer.train_with_early_stopping(model, train_data, val_data, optimizer)
```

### Hard Token収集とtrain/val分割

```python
# block_idx=0 でBlock 1の信頼度から閾値を計算
threshold = compute_confidence_threshold(model, val_data, target_ratio=0.5, device, block_idx=0)
# block_idx=0 でBlock 1のhidden statesを収集
hard_examples = collect_hard_examples(model, val_data, threshold, device, block_idx=0)

# Hard Examplesをtrain/valに分割（Phase 2以降の訓練用）
hard_train, hard_val = split_hard_examples(hard_examples, train_ratio=0.8)
```

### Block 2 追加・訓練

```python
# extend()でBlock追加
model_extended = model.extend(num_new_layers=2, threshold=threshold)
# blocks = [LEGOBlock(2層, threshold), LEGOBlock(2層, 1.0)]

# Hard Examples内でtrain/valが完結
# start_block_idx=1 は新しいBlock（Block 2）
result = trainer.train_block(
    model_extended, hard_train, hard_val, optimizer,
    start_block_idx=1
)
```

### 推論（TRUE Early Exit）

```python
generated, stats = model.generate(prompt, max_new_tokens=32)
# stats: {exit_counts: [block0_exits, block1_exits, ...], actual_compute_cost, shallow_ratio}
```

---

## 公開API

### LEGOBlock

| メソッド | 用途 |
|---------|------|
| `forward(h)` | 全レイヤー処理 |
| `forward_with_cache(h, cache)` | KVキャッシュ対応 |
| `compute_confidence(h)` | 信頼度計算（トークン単位） |
| `freeze()` / `unfreeze()` | パラメータ凍結 |

### LEGOTransformer

| メソッド | 用途 |
|---------|------|
| `create(...)` | 単一Blockモデル作成 |
| `forward(x)` | 標準推論（全ブロック通過） |
| `get_hidden_states(x, up_to_block)` | 指定ブロックまでのhidden states取得 |
| `forward_from_block(h, start_block_idx)` | 指定Blockから処理 |
| `forward_with_routing(x)` | TRUE Early Exit評価 |
| `generate(...)` | TRUE Early Exit生成 |
| `extend(...)` | Block追加 |

### Trainer

| メソッド | 用途 |
|---------|------|
| `train_with_early_stopping(...)` | Phase 1: 全データでBlock訓練 |
| `train_block(hard_train, hard_val, ..., start_block_idx)` | Phase 2+: Hard Examples内で完結する訓練 |
| `evaluate(model, val_batches, use_routing=False)` | 評価 |

### ユーティリティ

| 関数 | 用途 |
|------|------|
| `compute_confidence_threshold(..., block_idx)` | 閾値自動計算（block_idx必須） |
| `collect_hard_examples(..., block_idx)` | Hard Token収集（block_idx必須） |
| `split_hard_examples(hard_examples, train_ratio)` | Hard Examplesをtrain/valに分割 |
| `create_hard_example_loader()` | バッチ化 |

---

## ファイル構成

```
src/lego/
├── __init__.py      # 公開API
├── block.py         # LEGOBlock (nn.Module)
├── transformer.py   # LEGOTransformer (nn.Module)
├── trainer.py       # Trainer
├── utils.py         # ユーティリティ関数
├── config.py        # 設定
├── data.py          # データローダー
└── modules/         # 基本モジュール
    ├── attention.py
    ├── ffn.py
    ├── norm.py
    └── transformer.py  # TransformerBlock
```

---

## 設計原則

1. **LEGOBlockがレイヤーを所有** - 各Blockは自身のTransformerLayerを持つ
2. **LEGOTransformerは橋渡し役** - Block間のルーティングと管理
3. **start_block_idx / block_idx** - 常に明示的に指定（0-indexed）
4. **デフォルト引数値禁止** - block_idx等のブロック指定引数にはデフォルト値を設定しない
5. **Block訓練の独立性** - Phase 2以降の各Blockはtrain/valともにHard Examples内で完結。全データのval_batchesは使用しない
6. **トークン単位のEarly Exit** - すべての処理（訓練、評価、生成）でearly exitはトークン単位。バッチ単位ではない

---

## 注意事項

### KVキャッシュとEarly Exit

- Early exitした場合、後続BlockのKVキャッシュは**更新されない**（これが正しい動作）
- 例：Block 1でexit → Block 2のKVキャッシュは変更なし
- 次トークンでBlock 2まで処理が必要になれば、その時点で更新される

### RoPE

位置インデックスは累積位置を使用

### 核心機能（削除禁止）

1. `collect_hard_examples()` - トークン単位でhidden states収集
2. `forward_from_block()` - hidden statesから直接Block訓練
3. `compute_confidence()` - 信頼度計算（トークン単位）
