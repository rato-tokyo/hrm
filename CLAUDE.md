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

### LEGOBlock

```python
@dataclass
class LEGOBlock:
    start_layer: int      # 開始層（inclusive）
    end_layer: int        # 終了層（exclusive）
    threshold: float      # このBlock後のexit閾値（最終Blockは1.0）
```

---

## 使用例

### Block 1 訓練

```python
model = LEGOTransformer(vocab_size=vocab_size, dim=dim, num_layers=2)
# デフォルト: blocks = [LEGOBlock(0, 2, threshold=1.0)]

trainer = Trainer(vocab_size=vocab_size, device=device)
result = trainer.train_with_early_stopping(model, train_data, val_data, optimizer)
```

### Hard Token収集

```python
threshold = compute_confidence_threshold(model, val_data, target_ratio=0.5, device)
hard_tokens = collect_hard_examples(model, val_data, threshold, device)
```

### Block 2 追加・訓練

```python
# extend()でBlock追加
model_extended = model.extend(num_new_layers=2, threshold=threshold)
# blocks = [LEGOBlock(0, 2, threshold), LEGOBlock(2, 4, 1.0)]

result = trainer.train_upper_layers_with_early_stopping(
    model_extended, hard_batches, val_data, hard_tokens, optimizer,
    num_lower_layers=2
)
```

### Block 3 追加（オプション）

```python
# さらにBlock追加
model_3blocks = model_extended.extend(num_new_layers=2, threshold=new_threshold)
# blocks = [LEGOBlock(0, 2, threshold), LEGOBlock(2, 4, new_threshold), LEGOBlock(4, 6, 1.0)]
```

### 推論（TRUE Early Exit）

```python
generated, stats = model.generate(prompt, max_new_tokens=32)
# stats: {exit_counts: [block0_exits, block1_exits, ...], actual_compute_cost, ...}
```

---

## 公開API

### LEGOBlock

```python
LEGOBlock(start_layer, end_layer, threshold=1.0)
```

### LEGOTransformer

| メソッド | 用途 |
|---------|------|
| `forward(x)` | 標準推論（全Block通過） |
| `generate(...)` | 生成（TRUE Early Exit対応） |
| `forward_with_routing(x, threshold)` | ルーティング付き推論（評価用） |
| `forward_upper_layers(h, start)` | 指定層以降を処理 |
| `extend(num_new_layers, threshold)` | Block追加 |
| `compute_confidence(h)` | 信頼度計算 |

### Trainer

| メソッド | 用途 |
|---------|------|
| `train_with_early_stopping(...)` | Block訓練 |
| `train_upper_layers_with_early_stopping(...)` | 追加Block訓練 |
| `evaluate(...)` | 評価 |

### ユーティリティ

| 関数 | 用途 |
|------|------|
| `compute_confidence_threshold()` | 閾値自動計算 |
| `collect_hard_examples()` | Hard Token収集 |
| `create_hard_example_loader()` | バッチ化 |

---

## 開発ルール

### 変更時の必須手順

1. `python3 test_lego.py` で全テスト合格を確認

### 設計の注意点

1. **TrainerはModelのAPIのみ使用** - 内部構造への直接アクセス禁止
2. **LEGOBlock単位で考える** - 各Blockは独立した層の集合
3. **シンプルさ優先** - 複雑な抽象化より明確なコード

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

位置インデックスは累積位置を使用：
```python
cos, sin = self.rope(x, seq_len, position_offset=past_len)
```

### 核心機能（削除禁止）

1. `collect_hard_examples()` - トークン単位でhidden states収集
2. `forward_upper_layers()` - hidden statesから直接Block訓練
3. `compute_confidence()` - 信頼度計算
4. `LEGOBlock` - Block定義
