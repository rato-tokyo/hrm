# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## スコープ

**本フレームワークは事前学習（pre-training）専用です。**

- テキスト生成（generate）機能は含まない
- KVキャッシュは不要（事前学習では使わない）
- 推論は`forward_with_routing`で評価のみ

### Early ExitとKVキャッシュに関する設計方針

**本プロジェクトでは、Early Exitによって後続BlockのKVキャッシュが更新されないことを問題視しない。むしろ、Early Exit後も後続BlockのKVキャッシュが更新される実装こそが問題である。**

理由：
- Early Exitの目的は計算量削減。後続Blockを処理しないことで計算を省く
- 後続BlockのKVキャッシュが更新されるなら、その Block を通過したことになり、Early Exitの意味がない
- 「KVキャッシュが更新されない」は「そのBlockを通らなかった」の正しい結果

---

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

    def forward(h) -> h                                # 全レイヤー処理
    def forward_with_exit(h) -> (h, logits, exit_mask) # forward + exit判定
    def compute_confidence(h) -> (logits, confidence)  # 信頼度計算（トークン単位）
    def freeze() / unfreeze()                          # パラメータ凍結
```

### LEGOTransformer (nn.Module)

```python
class LEGOTransformer(nn.Module):
    """LEGOBlockの管理とルーティング。"""
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
logits, stats = model.forward_with_routing(x)
# stats: {exit_counts: [block0_exits, block1_exits, ...], compute_cost, shallow_ratio}
```

---

## 公開API

### LEGOBlock

| メソッド | 用途 |
|---------|------|
| `forward(h)` | 全レイヤー処理 |
| `forward_with_exit(h)` | forward + exit判定 |
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

1. **事前学習専用** - generate、KVキャッシュは不要。実装しない
2. **LEGOBlockがレイヤーとexit判定を所有** - 各Blockは自身のTransformerLayerを持ち、exit判定もBlockの責務
3. **LEGOTransformerはルーティングのみ** - Block間のインデックス管理と統計計算
4. **start_block_idx / block_idx** - 常に明示的に指定（0-indexed）
5. **デフォルト引数値禁止** - block_idx等のブロック指定引数にはデフォルト値を設定しない
6. **Block訓練の独立性** - Phase 2以降の各Blockはtrain/valともにHard Examples内で完結
7. **トークン単位のEarly Exit** - すべての処理でearly exitはトークン単位（バッチ単位ではない）

---

## TRUE Early Exitの実装

### 正しい実装パターン

```python
# forward_with_routing内のループ
active_indices = torch.arange(total_tokens)  # 全トークン

for block_idx, block in enumerate(self.blocks):
    if len(active_indices) == 0:
        break  # 全トークンexit済み → 後続blockをスキップ

    h_active = h_flat[active_indices]  # activeなトークンのみ取得

    # Blockがforward + exit判定を一括処理
    h_active, logits_active, exit_mask = block.forward_with_exit(h_active)

    if not is_last_block:
        # exitするトークンのlogitsを保存
        final_logits[active_indices[exit_mask]] = logits_active[exit_mask]
        # 継続するトークンのみに絞る（TRUE Early Exit）
        active_indices = active_indices[~exit_mask]
```

### 重要なポイント

- `active_indices`を更新することで、exitしたトークンは**後続blockを実際に通らない**
- Blockの`forward_with_exit`がexit判定を含むため、Transformerはインデックス管理に専念

---

## 注意事項

### RoPE

位置インデックスは累積位置を使用

### 核心機能（削除禁止）

1. `collect_hard_examples()` - トークン単位でhidden states収集
2. `forward_from_block()` - hidden statesから直接Block訓練
3. `compute_confidence()` - 信頼度計算（トークン単位）
4. `forward_with_exit()` - Block単位のexit判定
5. `forward_with_routing()` - TRUE Early Exit評価

---

## 過去の設計ミスと教訓

### 1. `forward_with_routing`の誤実装

**問題：**
評価時の`forward_with_routing`で、全トークンを全Blockに通してからマスクで統計を取る実装をしていた。

```python
# 誤った実装
for block in self.blocks:
    h = block(h)  # 全トークンが全Blockを通過
# 後からマスクで「どこでexitしたか」を記録するだけ
```

**教訓：** TRUE early exitを謳うなら、実際に計算をスキップしなければ意味がない。

### 2. 不要な機能の実装（generate、KVキャッシュ）

**問題：**
事前学習フレームワークなのに、テキスト生成用の`generate`メソッドとKVキャッシュ関連機能を実装していた。

**原因：**
- スコープを明確にしていなかった
- 「将来使うかもしれない」という曖昧な理由で実装
- 事前学習と推論/生成の区別が不明確だった

**教訓：**
- **スコープを最初に明確化する** - 何を作るのか、何を作らないのかを決める
- **YAGNI原則** - 今必要ないものは実装しない
- **事前学習専用と明記する** - 生成機能が必要なら別フレームワークで

### 3. exit判定がTransformerに散在

**問題：**
exit判定のロジック（`confidence >= threshold`）がTransformer側に書かれていた。

**修正：**
`forward_with_exit`をBlockに追加し、exit判定をBlock内に集約。

**教訓：** 責務を適切に分離する。Blockはexit判定、Transformerはルーティング。

### 4. 「保持する」を複雑に実装

**問題：**
「後続blockのKVキャッシュを保持する」処理を複雑に実装しようとした。

**正しい発想：**
「更新しない」は「何もしない」で実現できる。

**教訓：** 「Xを保持する」は「Xを変更しない」と同義。複雑な保持処理を書く前に、そもそも何もしなければいいことに気づくべき。

---

## 厳守事項

1. **事前学習専用** - generate、KVキャッシュは実装しない
2. **TRUE Early Exit** - exitしたトークンの後続blockは処理しない
3. **シンプルに保つ** - 不要な機能を追加しない
