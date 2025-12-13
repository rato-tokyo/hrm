# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## コア概念

LEGOは、2フェーズ訓練と**TRUE Early Exit**推論を組み合わせた効率的なTransformer訓練フレームワークです。

### 基本アイデア

1. **Phase 1**: 浅いモデル（2層）を全データで訓練
2. **Hard Example収集**: 信頼度の低いトークンを特定
3. **Phase 2**: 深いモデル（4層）の上位層のみをHard Examplesで訓練
4. **推論**: TRUE Early Exitで高信頼度トークンは上位層を**実際にスキップ**

### TRUE Early Exit vs Fake Early Exit

| 方式 | 動作 | 計算削減 |
|------|------|----------|
| **Fake** (旧設計) | 両パス計算→選択 | なし |
| **TRUE** (新設計) | 高信頼度→上位層スキップ | **実際に削減** |

実験結果: TRUE Early Exitで**32.8%の実計算削減**達成

---

## アーキテクチャ

### KVキャッシュ分離戦略

```python
# 下位層キャッシュ（常に更新）
lower_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]  # len = exit_layer

# 上位層キャッシュ（deepパス時のみ更新）
upper_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]  # len = num_layers - exit_layer
```

### Early Exit判定フロー

```
for each token:
    h = embedding(token)
    h = forward_lower_layers(h, lower_cache)  # Layer 0〜exit_layer-1

    confidence = compute_confidence(h)

    if confidence >= threshold:
        output = shallow_logits  # Layer exit_layer〜をスキップ
    else:
        h = forward_upper_layers(h, upper_cache)  # Layer exit_layer〜を実行
        output = deep_logits
```

---

## 信頼度とルーティング

### 信頼度（Confidence）

- **信頼度** = 予測確率の最大値（max probability）
- 実装: `F.softmax(logits, dim=-1).max(dim=-1).values`

### 閾値の決定

- **自動調整**: 指定比率のトークンがHard Examplesになるよう閾値を計算
- 実装: `compute_confidence_threshold(model, val_batches, target_ratio, device)`
- 例：`hard_example_ratio=0.5` → 信頼度の低い方から50%を「難しいトークン」とする閾値を自動算出

---

## 2フェーズ訓練の実装

### Phase 1: 浅いモデルの訓練

```python
model = LEGOTransformer(
    vocab_size=vocab_size, dim=dim, num_layers=2, num_heads=num_heads
)
trainer = Trainer(vocab_size=vocab_size, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
result = trainer.train_with_early_stopping(model, train_loader, val_loader, optimizer)
```

### Hard Examples収集

```python
confidence_threshold = compute_confidence_threshold(
    model, val_loader, target_ratio=0.5, device=device
)
hard_examples = collect_hard_examples(model, val_loader, confidence_threshold, device)
hard_batches = create_hard_example_loader(hard_examples, batch_size=64)
# Returns: {'hidden_states', 'targets'}
```

### Phase 2: 拡張モデルの訓練

```python
# extend メソッドで拡張（重みコピー＋凍結を自動化）
model_extended = model.extend(
    num_layers=4,
    routing_threshold=threshold,
    freeze_lower=True  # Layer 0-1 + embeddingを凍結
).to(device)

# Hard examplesで上位層のみ訓練
optimizer = torch.optim.AdamW(model_extended.parameters(), lr=1e-4)
result = trainer.train_upper_layers_with_early_stopping(
    model_extended, hard_batches, val_loader, hard_examples,
    optimizer, num_lower_layers=2,
    routing_threshold=threshold
)
```

### 推論: TRUE Early Exit

```python
# 生成時のTRUE Early Exit
generated, stats = model_extended.generate_with_early_exit(
    prompt, max_new_tokens=32, routing_threshold=threshold
)
# Returns: generated tokens, {shallow_count, deep_count, actual_compute_cost, shallow_ratio}
```

---

## 公開API

### LEGOTransformer

| メソッド | 用途 |
|---------|------|
| `forward(x)` | 標準推論（全層通過） |
| `forward_with_cache(x, past_kv_cache, use_cache)` | KVキャッシュ付き推論 |
| `forward_with_cache_partial(h, cache, start, end)` | 指定層範囲の処理 |
| `generate(input_ids, max_new_tokens, ...)` | 標準生成（KVキャッシュ使用） |
| `generate_with_early_exit(input_ids, max_new_tokens, routing_threshold, ...)` | **TRUE Early Exit生成** |
| `forward_upper_layers(h, start_layer)` | 上位層のみ処理（Phase 2訓練用） |
| `extend(num_layers, routing_threshold, freeze_lower)` | モデル拡張 |
| `compute_confidence(h)` | 信頼度計算 |
| `get_hidden_states(x)` | hidden states取得 |

### Trainer

| メソッド | 用途 |
|---------|------|
| `train_with_early_stopping(...)` | Phase 1 訓練 |
| `train_upper_layers_with_early_stopping(...)` | Phase 2 訓練 |
| `evaluate(model, val_batches, routing_threshold)` | 評価（Fake Early Exit統計） |

### ユーティリティ関数

| 関数 | 用途 |
|------|------|
| `set_seed()` | 再現性のためのシード設定 |
| `get_device()` | 利用可能なデバイス取得 |
| `create_synthetic_data()` | テスト用合成データ生成 |
| `compute_confidence_threshold()` | 指定比率で閾値を自動計算 |
| `collect_hard_examples()` | 閾値未満のトークンを収集 |
| `create_hard_example_loader()` | Hard examplesをバッチ化 |
| `evaluate_on_hard_examples()` | Hard examplesでの評価 |

---

## 実験スクリプト

| スクリプト | 内容 |
|-----------|------|
| `colab4.py` | **TRUE Early Exit検証**（2フェーズ訓練 + 生成） |

---

## 開発ルール

### 変更時の必須手順

1. `python3 test_lego.py` で全テスト合格を確認
2. `python3 -m mypy src/lego/ --ignore-missing-imports` でエラーなし
3. `python3 -m ruff check src/lego/` でエラーなし

### コード変更時のチェックリスト

- [ ] `test_lego.py` で全テスト合格
- [ ] mypy/ruff でエラーなし
- [ ] メソッドの外部インターフェースを不用意に変更していない
- [ ] 返り値の構造を変更する場合はテストも更新

### Git操作

変更完了後は以下を実行：

```bash
git add .
git commit -m "適切なコミットメッセージ"
git push origin main
```

---

## 教訓と注意事項

### 1. テスト閾値の設定

未訓練モデルでEarly Exitをテストする場合、閾値を低く設定する必要がある。
- 未訓練モデルの信頼度は低い（vocab_size=100で約1%）
- threshold=0.5では全トークンがdeepパスへ
- **threshold=0.02程度で適切にearly exitが発生**

### 2. KVキャッシュとRoPE

KVキャッシュ使用時、RoPEの位置インデックスは**累積位置**を使用する必要がある：
```python
# NG: 常に0から開始
cos, sin = self.rope(x, seq_len, position_offset=0)

# OK: 過去のキャッシュ長を考慮
cos, sin = self.rope(x, seq_len, position_offset=past_len)
```

### 3. 削除禁止の核心機能

以下はLEGOの効率性を実現する核心機能：
1. `collect_hard_examples()` - トークン単位でhidden statesを収集
2. `create_hard_example_loader()` - hidden statesをバッチ化
3. `forward_upper_layers()` - hidden statesから直接上位層を訓練
4. `evaluate_on_hard_examples()` - hard examplesのPPL計算

### 4. Gitマージコンフリクト防止

大規模変更時は事前にリモート状態を確認：
```bash
git fetch origin
git log --oneline --graph origin/main -10
git diff HEAD origin/main --stat
```

### 5. Human-AI タスク分担

- **Human**: 重い環境セットアップ、GUI操作、外部サービス認証
- **AI**: コード作成・修正、git操作、ドキュメント作成
