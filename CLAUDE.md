# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## コア概念

LEGOは、**Block単位の段階的訓練**と**TRUE Early Exit**推論を組み合わせた効率的なTransformer訓練フレームワークです。

### 基本アイデア（Block-based設計）

```
Block 1 訓練 → Hard Tokens出力 → Block 2 訓練 → Hard Tokens出力 → ...
```

1. **Block 1**: 最初のブロックを全データで訓練
2. **Hard Token収集**: 訓練後、信頼度の低いトークンを自動出力
3. **Block 2**: 次のブロックをHard Tokensのみで訓練
4. **繰り返し**: 必要に応じてBlock 3, 4...と拡張可能
5. **推論**: TRUE Early Exitで高信頼度トークンは後続Blockを**実際にスキップ**

### TRUE Early Exit

| 方式 | 動作 | 計算削減 |
|------|------|----------|
| **Fake** | 全Block計算→選択 | なし |
| **TRUE** | 高信頼度→後続Blockスキップ | **実際に削減** |

---

## 設計原則

### 1. シンプルなBlock構造

- 各Blockは独立した層の集合
- Block間の境界で信頼度を評価
- 信頼度が高ければ後続Blockをスキップ

### 2. 責務の分離

| コンポーネント | 責務 |
|---------------|------|
| **Model** | 推論（forward, generate）、信頼度計算、ルーティング |
| **Trainer** | 訓練ループ、評価（Modelのメソッドを呼ぶだけ） |
| **Utils** | Hard Token収集、データ処理 |

**重要**: TrainerはModelの内部構造（layers, embedding）に直接アクセスしない

### 3. 信頼度とルーティング

- **信頼度** = 予測確率の最大値（max probability）
- **閾値の自動決定**: 指定比率のトークンがHard Tokensになるよう閾値を計算
- **ルーティング**: `confidence >= threshold` → 現在のBlock出力を使用

---

## 使用例

### Block 1 訓練

```python
model = LEGOTransformer(vocab_size=vocab_size, dim=dim, num_layers=2)
trainer = Trainer(vocab_size=vocab_size, device=device)
result = trainer.train_with_early_stopping(model, train_data, val_data, optimizer)
```

### Hard Token収集（自動）

```python
threshold = compute_confidence_threshold(model, val_data, target_ratio=0.5, device)
hard_tokens = collect_hard_examples(model, val_data, threshold, device)
# Returns: {'hidden_states', 'targets'}
```

### Block 2 訓練

```python
model_extended = model.extend(num_layers=4, routing_threshold=threshold)
# Block 1の重みはコピー＆凍結、Block 2は新規初期化

result = trainer.train_upper_layers_with_early_stopping(
    model_extended, hard_batches, val_data, hard_tokens, optimizer,
    num_lower_layers=2, routing_threshold=threshold
)
```

### 推論（TRUE Early Exit）

```python
generated, stats = model.generate(prompt, max_new_tokens=32, routing_threshold=threshold)
# stats: {shallow_count, deep_count, actual_compute_cost, shallow_ratio}
```

---

## 公開API

### LEGOTransformer

| メソッド | 用途 |
|---------|------|
| `forward(x)` | 標準推論（全Block通過） |
| `generate(...)` | 生成（TRUE Early Exit対応） |
| `forward_with_routing(x, threshold)` | ルーティング付き推論（評価用） |
| `forward_upper_layers(h, start)` | 指定Block以降を処理 |
| `extend(num_layers, threshold)` | Block追加 |
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
2. **Block単位で考える** - lower/upperではなくBlock 1, Block 2
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
cos, sin = self.rope(x, seq_len, position_offset=past_len)  # OK
```

### 核心機能（削除禁止）

1. `collect_hard_examples()` - トークン単位でhidden states収集
2. `forward_upper_layers()` - hidden statesから直接Block訓練
3. `compute_confidence()` - 信頼度計算
