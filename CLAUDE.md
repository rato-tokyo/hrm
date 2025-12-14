# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## スコープ

**本フレームワークは事前学習（pre-training）専用です。**

- テキスト生成（generate）機能は含まない
- KVキャッシュは実装しない（事前学習では不要）
- 推論は`forward(return_stats=True)`で評価

---

## アーキテクチャ

```
TransformerLayer    → 1層（Attention + FFN）
TransformerBlock    → 複数層のスタック（標準Transformer）
LEGOBlock           → TransformerBlock + early exit機能
LEGOLLM             → LEGOBlock × N（モデル全体）
train_block()       → Block訓練関数（外部）
```

### ファイル構成

```
lego/
├── modules/
│   ├── transformer.py  # TransformerLayer, TransformerBlock
│   ├── attention.py    # MultiHeadAttention
│   ├── ffn.py          # GatedLinearUnit
│   └── norm.py         # RMSNorm
├── block.py            # LEGOBlock（推論のみ、約90行）
├── model.py            # LEGOLLM
├── trainer.py          # train_block()（訓練ロジック）
├── data.py             # SequenceData
└── config.py           # ExperimentConfig, TrainerConfig
```

---

## コア概念

LEGOは、**LEGOBlock単位の段階的訓練**と**TRUE Early Exit**推論を組み合わせた効率的なTransformer訓練フレームワークです。

### 訓練フロー

1. **Block 0**: 最初のブロックを全データで訓練
2. **Hard Sequence収集**: 訓練後、信頼度の低いトークンを含むシーケンスを自動出力
3. **Block 1+**: Hard Sequencesのみで訓練
4. **推論**: TRUE Early Exitで高信頼度トークンは後続Blockを**実際にスキップ**

---

## 設計原則

1. **事前学習専用** - generate、KVキャッシュは実装しない
2. **コンポジション方式** - LEGOBlockはTransformerBlockをラップ（継承ではない）
3. **LEGOBlockがexit判定を所有** - 各Blockはthresholdを持ち、softmax maxで信頼度計算
4. **LEGOLLMはルーティングのみ** - Block間のインデックス管理と統計計算
5. **トークン単位のEarly Exit** - exit判定はトークン単位（バッチ単位ではない）
6. **TRUE Early Exit** - exitしたトークンの後続blockは処理しない
7. **訓練と推論の分離** - LEGOBlockは推論のみ、訓練は`train_block()`関数
8. **デフォルト値禁止** - 関数・クラスの引数にデフォルト値を設定しない（意図しない動作の原因）
9. **シーケンス単位処理** - Attention計算のためシーケンス全体を処理、exit判定のみトークン単位

---

## 核心機能（削除禁止）

1. `LEGOBlock.forward()` - Transformer処理 + exit判定（h, logits, should_exit）
2. `train_block()` - Block訓練 + hard example収集（trainer.py）
3. `LEGOLLM.forward()` - TRUE Early Exit推論
4. `SequenceData` - hidden states + targetsのコンテナ（シーケンス単位）

### LEGOBlockの責務（シンプル）

```python
# LEGOBlockはTransformerBlockを引数で受け取る（明示的コンポジション）
block = LEGOBlock(TransformerBlock(dim=256, num_heads=8, num_layers=4))

class LEGOBlock(nn.Module):
    def __init__(self, transformer: TransformerBlock):
        self.transformer = transformer  # 外部から注入
        self.threshold = 1.0  # trainerが設定

    # プロパティ（transformerに委譲）
    @property
    def dim(self) -> int: return self.transformer.dim
    @property
    def num_layers(self) -> int: return self.transformer.num_layers

    # メソッド
    forward() → (h, logits, should_exit)  # 推論のみ
    set_output_head()                      # 共有出力層の設定
```

### 信頼度計算方式（重要：削除禁止）

**softmax max方式を使用する**：

```python
# 正しい実装（softmax max方式）
confidence = F.softmax(logits, dim=-1).max(dim=-1).values
```

- 訓練完了後の言語モデル出力に基づく信頼度
- 追加パラメータ不要
- 訓練は言語モデリング損失のみ

### Hard Example収集方式（重要：削除禁止）

**シーケンス単位で収集**：`hard_ratio=0.5`なら信頼度下位50%のトークンを含むシーケンスをhard exampleとして収集。

```python
# 正しい実装（シーケンス単位）
# 1. 全トークンのconfidenceからthresholdを計算
threshold = torch.quantile(all_confidences_flat, 1.0 - hard_ratio)

# 2. 各シーケンス内の最小confidenceがthreshold未満なら、そのシーケンスはhard
min_confidence_per_seq = confidences.min(dim=1).values
hard_sequence_mask = min_confidence_per_seq < threshold

# 3. hard sequenceのみを返す（Block 0の出力hidden statesとtargets）
return SequenceData(hidden_out[hard_sequence_mask], targets[hard_sequence_mask])
```

### Threshold自動設定方式（重要：削除禁止）

**thresholdは`train_block()`内で自動計算される**：外部からハードコードしない。

```python
# 正しい実装（quantile方式）
# hard_ratio=0.5なら、上位50%がexitするthresholdを計算
threshold = torch.quantile(all_confidences, 1.0 - hard_ratio)
block.threshold = threshold
```

これにより：
- 訓練後のsoftmax出力分布に基づいた適切なthreshold
- `hard_ratio`と推論時のexit率が一致
- 外部での手動調整が不要

---

## シーケンス単位処理の重要性（重要：削除禁止）

**Attentionはシーケンス全体を必要とする**：

```python
# 正しい実装（シーケンス単位）
h = block.forward(sequences)  # (batch, seq_len, dim)
# Attention: 各トークンが他のトークンを参照可能

# 間違った実装（トークン単位）
h = block.forward(tokens)  # (batch, 1, dim)
# Attention: 各トークンが自分自身のみ参照 → 文脈なし
```

訓練時も推論時も、シーケンス全体を処理してからexit判定を行う。

---

## 過去の設計ミスと教訓

### 1. TRUE Early Exitの誤実装

**問題：** 全トークンを全Blockに通してからマスクで統計を取る実装をしていた。

**教訓：** TRUE early exitを謳うなら、実際に計算をスキップしなければ意味がない。

### 2. 不要な機能の実装（generate、KVキャッシュ）

**問題：** 事前学習フレームワークなのに生成機能を実装していた。

**教訓：** スコープを最初に明確化する。YAGNI原則。

### 3. exit判定がTransformerに散在

**問題：** exit判定のロジックがTransformer側に書かれていた。

**教訓：** 責務を適切に分離する。Blockはexit判定、Transformerはルーティング。

### 4. 「保持する」を複雑に実装

**問題：** 「後続blockのKVキャッシュを保持する」処理を複雑に実装しようとした。

**教訓：** 「Xを保持する」は「Xを変更しない」と同義。何もしなければいい。

### 5. モデルに訓練ロジックを含める

**問題：** LEGOBlockに`fit()`メソッドを実装し、300行超のクラスになっていた。

**教訓：** 訓練と推論を分離する。モデルは推論のみ、訓練は外部関数で。

### 6. トークン単位でAttention処理

**問題：** 訓練時にトークンを独立して `(batch, 1, dim)` で処理し、Attentionが機能していなかった。

**教訓：** Attentionはシーケンス全体を必要とする。シーケンス単位で処理し、exit判定のみトークン単位にする。
