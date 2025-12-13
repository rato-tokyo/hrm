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
    """複数のTransformerLayerを持つブロック。最終層でEarly Exit判定。"""
    def __init__(self, dim: int, num_heads: int, num_layers: int, threshold: float = 1.0):
        self.layers = nn.ModuleList([...])  # TransformerBlockのリスト
        self.threshold = threshold          # Early Exit閾値

    def forward(self, h) -> h                    # 全レイヤー処理
    def forward_with_cache(h, cache) -> (h, cache)  # KVキャッシュ対応
    def should_exit(confidence) -> bool          # Early Exit判定
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

    def forward(x) -> logits                     # 標準推論
    def forward_from_block(h, start_block_idx)   # 指定Blockから処理
    def forward_with_routing(x) -> (logits, stats)  # ルーティング評価
    def generate(...) -> (tokens, stats)         # TRUE Early Exit生成
    def extend(num_new_layers, threshold) -> LEGOTransformer  # Block追加
    def compute_confidence(h) -> (logits, confidence)
```

---

## 使用例

### Block 1 訓練

```python
model = LEGOTransformer.create(vocab_size=vocab_size, dim=dim, num_layers=2, num_heads=4)

trainer = Trainer(vocab_size=vocab_size, device=device)
result = trainer.train_with_early_stopping(model, train_data, val_data, optimizer)
```

### Hard Token収集

```python
# block_idx=0 でBlock 1の信頼度から閾値を計算
threshold = compute_confidence_threshold(model, val_data, target_ratio=0.5, device, block_idx=0)
# block_idx=0 でBlock 1のhidden statesを収集
hard_tokens = collect_hard_examples(model, val_data, threshold, device, block_idx=0)
```

### Block 2 追加・訓練

```python
# extend()でBlock追加
model_extended = model.extend(num_new_layers=2, threshold=threshold)
# blocks = [LEGOBlock(2層, threshold), LEGOBlock(2層, 1.0)]

# start_block_idx=1 は新しいBlock（Block 2）
result = trainer.train_new_block_with_early_stopping(
    model_extended, hard_batches, val_data, hard_tokens, optimizer,
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
| `forward_with_routing(h)` | ルーティング判定付きforward |
| `compute_confidence(h)` | 信頼度計算（output_head共有） |
| `should_exit(confidence)` | Early Exit判定 |
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
| `train_with_early_stopping(...)` | Block訓練 |
| `train_new_block_with_early_stopping(..., start_block_idx)` | 新Block訓練 |
| `evaluate(model, val_batches, use_routing=False)` | 評価 |

### ユーティリティ

| 関数 | 用途 |
|------|------|
| `compute_confidence_threshold(..., block_idx)` | 閾値自動計算（block_idx必須） |
| `collect_hard_examples(..., block_idx)` | Hard Token収集（block_idx必須） |
| `create_hard_example_loader()` | バッチ化 |
| `train_new_block(..., start_block_idx)` | 新Block訓練（1エポック） |
| `evaluate_on_hard_examples(..., start_block_idx)` | Hard例評価 |

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

## 開発ルール

### 変更時の必須手順

1. `python3 test_lego.py` で全テスト合格を確認

### 設計原則

1. **LEGOBlockがレイヤーを所有** - 各Blockは自身のTransformerLayerを持つ
2. **LEGOTransformerは橋渡し役** - Block間のルーティングと管理
3. **start_block_idx / block_idx** - 常に明示的に指定（0-indexed）

### デフォルト引数値の禁止

**⚠️ 重要**: 関数やメソッドにおいて、`block_idx`、`start_block_idx`、`up_to_block`などのブロック指定引数には**デフォルト値を設定しない**こと。

**理由**:
- デフォルト値は暗黙の仮定を作り、予期せぬ動作の原因となる
- 3ブロック以上の場合、どのブロックを対象とするか明示すべき
- 呼び出し側で常に意図を明確にすることでバグを防ぐ

**禁止例**:
```python
def compute_confidence_threshold(..., block_idx: int = 0):  # NG
def collect_hard_examples(..., block_idx: int = 0):  # NG
def get_hidden_states(x, up_to_block: int = 0):  # NG
```

**正しい例**:
```python
def compute_confidence_threshold(..., block_idx: int):  # OK
def collect_hard_examples(..., block_idx: int):  # OK
def get_hidden_states(x, up_to_block: int):  # OK
```

### Git操作

```bash
git add .
git commit -m "適切なコミットメッセージ"
git push origin main
```

---

## 注意事項

### テスト閾値

未訓練モデルは信頼度が低いため、テスト時は低い閾値（0.02程度）を使用

### KVキャッシュとRoPE

位置インデックスは累積位置を使用

### 核心機能（削除禁止）

1. `collect_hard_examples()` - トークン単位でhidden states収集
2. `forward_from_block()` - hidden statesから直接Block訓練
3. `compute_confidence()` - 信頼度計算
4. `LEGOBlock.should_exit()` - Early Exit判定
