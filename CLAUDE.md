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
├── data.py             # TrainingData
└── config.py           # ExperimentConfig, TrainerConfig
```

---

## コア概念

LEGOは、**LEGOBlock単位の段階的訓練**と**TRUE Early Exit**推論を組み合わせた効率的なTransformer訓練フレームワークです。

### 訓練フロー

1. **Block 0**: 最初のブロックを全データで訓練
2. **Hard Token収集**: 訓練後、信頼度の低いトークンを自動出力
3. **Block 1+**: Hard Tokensのみで訓練
4. **推論**: TRUE Early Exitで高信頼度トークンは後続Blockを**実際にスキップ**

---

## 設計原則

1. **事前学習専用** - generate、KVキャッシュは実装しない
2. **コンポジション方式** - LEGOBlockはTransformerBlockをラップ（継承ではない）
3. **LEGOBlockがexit判定を所有** - 各Blockはthresholdを持ち、softmax maxで信頼度計算
4. **LEGOLLMはルーティングのみ** - Block間のインデックス管理と統計計算
5. **トークン単位のEarly Exit** - すべての処理でearly exitはトークン単位（バッチ単位ではない）
6. **TRUE Early Exit** - exitしたトークンの後続blockは処理しない
7. **訓練と推論の分離** - LEGOBlockは推論のみ、訓練は`train_block()`関数
8. **デフォルト値禁止** - 関数・クラスの引数にデフォルト値を設定しない（意図しない動作の原因）

---

## 核心機能（削除禁止）

1. `LEGOBlock.forward()` - Transformer処理 + exit判定（h, logits, should_exit）
2. `train_block()` - Block訓練 + hard example収集（trainer.py）
3. `LEGOLLM.forward()` - TRUE Early Exit推論
4. `TrainingData` - hidden states + targetsのコンテナ

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

**ratio方式を使用する**：`hard_ratio=0.5`なら信頼度下位50%のトークンをhard exampleとして収集。

```python
# 正しい実装（ratio方式）
num_hard = int(len(all_confidences) * hard_ratio)
_, hard_indices = torch.topk(all_confidences, num_hard, largest=False)
```

threshold方式（`confidence < threshold`）ではない。ratio方式は訓練データ量を制御可能にする。

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
