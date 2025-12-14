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
├── block.py            # LEGOBlock（推論のみ）
├── exit_classifier.py  # ExitClassifier（信頼度計算・exit判定）
├── model.py            # LEGOLLM（推論のみ）
├── model_trainer.py    # train_legollm(), evaluate_legollm()（LEGOLLM訓練・評価）
├── trainer.py          # train_block(), _train_lm()（LEGOBlock訓練）
├── exit_trainer.py     # train_exit_classifier(), collect_hard_examples()
├── data.py             # SequenceData
└── config.py           # ExperimentConfig, TrainerConfig
```

---

## コア概念

LEGOは、**LEGOBlock単位の段階的訓練**と**TRUE Early Exit**推論を組み合わせた効率的なTransformer訓練フレームワークです。

### 訓練フロー

1. **Block 0**: 最初のブロックを全データで訓練
2. **exit_classifier訓練**: LM訓練後、exit_classifierを訓練（loss方式）
3. **Hard Sequence収集**: 信頼度の低いトークンを含むシーケンスを自動出力
4. **Block 1+**: Hard Sequencesのみで訓練
5. **推論**: TRUE Early Exitで高信頼度トークンは後続Blockを**実際にスキップ**

### train_block()の内部フロー（重要）

**exit_classifierの訓練はLM訓練完了後に行う**：

```
train_block()
├── 1. データ分割 (train/val)
├── 2. _train_lm()                          ← Transformer + output_head の訓練（early stopping付き）
├── 3. block.forward()で全データを処理       ← hidden_states, logits を取得
├── 4. train_exit_classifier()              ← exit_classifier のみ訓練（hidden_states, logits, targets を渡す）
├── 5. collect_hard_examples()              ← threshold設定 + hard example収集
└── 6. 統計をまとめてreturn
```

**この順序の理由**：
- exit_classifierの訓練ラベルはLMの出力（logits）に依存する
- `exit_labels = torch.exp(-cross_entropy_loss)` を計算するため、LMが収束してからでないと適切なラベルが得られない
- train_exit_classifier()はLEGOBlockの内部構造を知らない（hidden_states, logits, targetsのみ受け取る）

### exit_trainer.pyの設計（重要）

**exit_trainerはLEGOBlockに依存しない**：

```python
# train_exit_classifier()の引数
train_exit_classifier(
    exit_classifier,   # 訓練対象
    hidden_states,     # block.forward()の出力
    exit_labels,       # 事前計算済みラベル（exp(-loss)）
    lr, num_epochs, is_verbose
)

# collect_hard_examples()の引数
collect_hard_examples(
    exit_classifier,   # 訓練済みExitClassifier
    hidden_states,     # block.forward()の出力
    targets,           # 正解ラベル
    seq_len,           # シーケンス長
    hard_ratio         # hard example比率
)
```

**この設計の利点**：
- exit_trainerはLEGOBlockの内部構造を知らない
- ExitClassifierを単体でテスト可能
- 別のモデル構造でもlogitsさえあれば使える

---

## 設計原則

1. **事前学習専用** - generate、KVキャッシュは実装しない
2. **コンポジション方式** - LEGOBlockはTransformerBlockをラップ（継承ではない）
3. **LEGOBlockがexit判定を所有** - 各Blockはthresholdを持ち、exit_classifierで信頼度計算
4. **LEGOLLMはルーティングのみ** - Block間のインデックス管理と統計計算
5. **トークン単位のEarly Exit** - exit判定はトークン単位（バッチ単位ではない）
6. **TRUE Early Exit** - exitしたトークンの後続blockは処理しない
7. **訓練と推論の分離** - LEGOBlockは推論のみ、訓練は`train_block()`関数
8. **デフォルト値禁止** - 関数・クラスの引数にデフォルト値を設定しない（意図しない動作の原因）
9. **シーケンス単位処理** - Attention計算のためシーケンス全体を処理、exit判定のみトークン単位

---

## 核心機能（削除禁止）

1. `LEGOBlock.forward()` - Transformer処理 + exit判定（h, logits, should_exit）
2. `LEGOLLM.forward()` - TRUE Early Exit推論
3. `train_legollm()` - LEGOLLM全体の訓練（model_trainer.py）
4. `train_block()` - LEGOBlock訓練 + exit_classifier訓練 + hard example収集（trainer.py）
5. `evaluate_legollm()` - LEGOLLM評価（model_trainer.py）
6. `SequenceData` - hidden states + targetsのコンテナ（シーケンス単位）

### LEGOBlockの責務（シンプル）

```python
# LEGOBlockはTransformerBlockを引数で受け取る（明示的コンポジション）
block = LEGOBlock(TransformerBlock(dim=256, num_heads=8, num_layers=4, ...))

class LEGOBlock(nn.Module):
    def __init__(self, transformer: TransformerBlock):
        self.transformer = transformer           # 外部から注入
        self.exit_classifier = ExitClassifier(transformer.dim)  # 信頼度計算用
        self.output_head: nn.Linear | None = None  # LEGOLLMが設定

    # プロパティ（transformerに委譲）
    @property
    def dim(self) -> int: return self.transformer.dim
    @property
    def threshold(self) -> float: return self.exit_classifier.threshold

    # メソッド
    forward() → (h, logits, should_exit)  # 推論のみ
    set_output_head()                      # 共有出力層の設定
```

### ExitClassifierの責務

```python
class ExitClassifier(nn.Module):
    def __init__(self, dim: int):
        self.linear = nn.Linear(dim, 1)
        self.threshold = 1.0  # trainerが設定

    # メソッド
    forward(h) → (confidence, should_exit)
    compute_confidence(h) → confidence
```

---

## 信頼度計算方式（重要：削除禁止）

**exit_classifier + loss方式を使用する**：

```python
# BDR-style: exit_classifierはlossを直接予測（sigmoidなし）
predicted_loss = block.exit_classifier.compute_confidence(h)
# 内部: self.linear(h).squeeze(-1)  # 生の出力

# exit_classifierの訓練ラベル（LM訓練完了後）- exit_trainer.pyで使用
per_token_loss = F.cross_entropy(logits, y, reduction='none')
exit_labels = per_token_loss  # lossそのもの（exp(-loss)ではない）
```

### BDR-style方式の理由

**問題**: 以前の `exp(-loss)` 方式では、ラベルがほぼ0に集中（mean=0.06）し、
sigmoid出力（0.5付近）との乖離が大きく、学習が機能しなかった。

**解決**: BDR（Bimodal Distribution Removal）研究に倣い、lossを直接予測。
- **低い predicted_loss = easy token = early exit**
- **高い predicted_loss = hard token = 次のBlockへ**

### 訓練フロー

1. **LM訓練**: 言語モデリング損失でTransformerを訓練（early stopping付き）
2. **exit_classifier訓練**: LM訓練完了後、hidden_statesからlossを予測するよう訓練
3. **threshold設定**: predicted_lossの分布からquantileでthresholdを計算

---

## Hard Example収集方式（重要：削除禁止）

**トークン単位で収集**：`hard_ratio=0.5`ならpredicted_loss上位50%のトークンのみをhard exampleとして収集。

```python
# 正しい実装（トークン単位）- exit_trainer.py の collect_hard_examples()
# 1. exit_classifierでpredicted_lossを計算
predicted_loss = block.exit_classifier.compute_confidence(h_out)

# 2. thresholdを計算（上位hard_ratio%がhard）
# high predicted_loss = hard token
threshold = torch.quantile(all_preds_flat, 1.0 - hard_ratio)

# 3. 各トークンがhardかどうか判定
hard_token_mask = predicted_loss > threshold  # (num_sequences, seq_len)

# 4. hardトークンのみを抽出
hard_hidden = hidden_out[hard_token_mask]  # (num_hard_tokens, dim)
hard_targets = targets[hard_token_mask]    # (num_hard_tokens,)

# 5. 新しいシーケンスに再構成してBlock 1に渡す
hard_hidden = hard_hidden.view(-1, seq_len, dim)
hard_targets = hard_targets.view(-1, seq_len)
return SequenceData(hard_hidden, hard_targets)
```

**重要**: Block 1が受け取るのはhardトークンのみ。easyトークンのhidden statesはBlock 1に流れない。

---

## ⛔ シーケンス単位収集は使用禁止

### 定義

| 用語 | 対象 | 選択基準 | 結果 |
|------|------|----------|------|
| **トークン単位収集** | 個々のトークン | 各トークンのconfidence < threshold | hardトークンのみ抽出、repackして新シーケンス作成 |
| **シーケンス単位収集** | シーケンス全体 | シーケンス内の最小confidenceで判定 | hardシーケンス全体を選択（easyトークンも含む） |

### シーケンス単位収集の詳細（⚠️ 使用禁止）

```python
# シーケンス単位収集（使用禁止）
# 対象: シーケンス全体
# 選択基準: シーケンス内の最小confidence（= 最も難しいトークンのconfidence）
min_confidence_per_seq = confidences.min(dim=-1).values  # (num_sequences,)
hard_seq_mask = min_confidence_per_seq < threshold  # (num_sequences,)

# 結果: 難しいトークンを1つでも含むシーケンス全体を選択
hard_hidden = hidden_out[hard_seq_mask]  # シーケンス全体（easyトークンも含む）
hard_targets = targets[hard_seq_mask]
```

### 禁止理由

1. **easyトークンがBlock 1に流れる**: シーケンス内の簡単なトークンも一緒に訓練される
2. **訓練効率の低下**: Block 0で十分処理できるトークンを再度訓練する無駄
3. **LEGOの設計思想に反する**: 「難しいものだけを深い層で処理」が核心

### 永続的な方針

**トークン単位収集のみを使用する**。シーケンス単位収集は今後一切実装しない。

文脈が失われる問題は、repack後のシーケンスでAttentionが新たな文脈を構築することで対処する。
元のシーケンス境界を維持する必要はない。

### Threshold自動設定方式（重要：削除禁止）

**thresholdは`train_block()`内で自動計算される**：外部からハードコードしない。

```python
# 正しい実装（quantile方式）
# hard_ratio=0.5なら、下位50%がhardトークン
threshold = torch.quantile(all_confidences, hard_ratio)
block.threshold = threshold
```

これにより：
- exit_classifierの出力分布に基づいた適切なthreshold
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

### 7. Hard example収集でシーケンス全体を渡す

**問題：** 「hardトークンを含むシーケンス全体」をBlock 1に渡し、easyトークンも含めて訓練していた。

**教訓：** Block 1が受け取るべきはhardトークンのみ。easyトークンのhidden statesはBlock 1に流れてはいけない。

### 8. 複数の信頼度計算方式を保持

**問題：** softmax方式、exit_classifier + correct/distill/loss など複数の方式をオプションとして保持していた。

**教訓：** 実験で最良と判明した方式（exit_classifier + loss）に一本化する。メンテナンス性 > 柔軟性。

### 9. 訓練データでvalidation PPLを計算

**問題：** `train_block()`内で訓練データを80/20分割し、20%側でval_pplを計算していた。結果として「val_ppl」は訓練データの一部の性能であり、真のvalidationセットの性能ではなかった。

**発覚経緯：** サニティチェック（Final PPL > worst block val_ppl）が失敗し、調査の結果バグと判明。

**解決：** `train_block()`と`train_legollm()`に別々の`train_data`と`val_data`を渡す設計に変更。WikiTextのvalidationセットを独立して使用するようにした。

**教訓：** 訓練データとvalidationデータは完全に分離する。内部分割は混乱の元。

### 10. 巨大なlogitsテンソルをメモリに保持

**問題：** exit_classifier訓練のため、全データのlogits `(num_sequences, seq_len, vocab_size)` をメモリに保持。vocab_size=50000の場合、~40GBのRAMを消費していた。

**解決：** logitsを保持せず、各バッチで即座に`exit_labels = exp(-loss)`を計算してCPUに移動。train_exit_classifierの引数を`(hidden_states, logits, targets)`から`(hidden_states, exit_labels)`に変更。

**教訓：** 巨大な中間テンソルは保持せず、必要な値（ここではexit_labels）だけを計算して保存する。

---

## 頻発する設計ミス（⚠️ 要注意）

以下のミスは繰り返し発生している。実装時に必ず確認すること。

### 1. シーケンス単位処理とトークン単位early exitの混同

**原則**：
- **Attention計算** → シーケンス単位（`(batch, seq_len, dim)`）
- **Early exit判定** → トークン単位（各トークンが独立してexit）
- **Hard example収集** → トークン単位（hardトークンのみ抽出）
- **訓練時のforward** → シーケンス単位（Attentionのため）

**よくあるミス**：
- Attentionをトークン単位で処理（文脈なし）
- Hard exampleをシーケンス単位で収集（easyトークンも含む）
- Exit判定をシーケンス単位で行う（全トークン一律）

### 2. Block間のデータフローの誤解

**正しいフロー**：
```
Block 0入力: embedding(tokens)           → (batch, seq_len, dim)
Block 0出力: hidden_states               → (batch, seq_len, dim)
Hard収集:    hardトークンのみ抽出         → (num_hard, dim)
再構成:      新シーケンスに詰め直し       → (new_batch, seq_len, dim)
Block 1入力: 再構成されたhardトークン     → (new_batch, seq_len, dim)
```

**よくあるミス**：
- Block 1にeasyトークンも渡す
- シーケンス境界を維持しようとする（不要）
