# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## スコープ

**本フレームワークは事前学習（pre-training）専用です。**

- テキスト生成（generate）機能は含まない
- KVキャッシュは実装しない（事前学習では不要）
- 推論は`forward(return_stats=True)`で評価

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
2. **LEGOBlockがレイヤーとexit判定を所有** - 各Blockは自身のTransformerLayerを持ち、exit判定もBlockの責務
3. **LEGOTransformerはルーティングのみ** - Block間のインデックス管理と統計計算
4. **トークン単位のEarly Exit** - すべての処理でearly exitはトークン単位（バッチ単位ではない）
5. **TRUE Early Exit** - exitしたトークンの後続blockは処理しない

---

## 核心機能（削除禁止）

1. `LEGOBlock.forward()` - レイヤー処理 + exit判定（h, logits, should_exit）
2. `LEGOBlock.train()` - Block訓練 + hard example収集
3. `LEGOTransformer.forward()` - TRUE Early Exit推論
4. `TrainingData` - hidden states + targetsのコンテナ

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
