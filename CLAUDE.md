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
TransformerBlock    → 複数層のスタック（hidden_historyを返す）
LEGOBlock           → TransformerBlock + exit_fn による早期exit判定
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
├── block.py            # LEGOBlock, default_exit_fn, ExitFn
├── model.py            # LEGOLLM（推論のみ）
├── model_trainer.py    # train_legollm(), evaluate_legollm()
├── trainer.py          # train_block(), _train_lm()
├── data.py             # SequenceData
└── config.py           # ExperimentConfig, TrainerConfig
```

---

## コア概念

LEGOは、**LEGOBlock単位の段階的訓練**と**TRUE Early Exit**推論を組み合わせた効率的なTransformer訓練フレームワークです。

### 訓練フロー

1. **Block 0**: 最初のブロックを全データで訓練
2. **hidden_history取得**: forward後、各レイヤーのhidden statesリストを取得
3. **cos_sim計算**: hidden_history[-2]とhidden_history[-1]のcos_simでthreshold設定
4. **Hard Sequence収集**: cos_simが低い（変化が大きい）トークンを収集
5. **Block 1+**: Hard Sequencesのみで訓練
6. **推論**: TRUE Early Exitで高cos_simトークンは後続Blockを**実際にスキップ**

### train_block()の内部フロー

```
train_block()
├── 1. _train_lm()                     ← Transformer + output_head の訓練（early stopping付き）
├── 2. hidden_history取得              ← block.forward()で各レイヤー出力を取得
├── 3. cos_sim計算                     ← hidden_history[-2], [-1]のcos_sim
├── 4. threshold設定                    ← hard_ratio quantileで設定
├── 5. hard example収集                 ← cos_sim < threshold のトークンを抽出
└── 6. 統計をまとめてreturn
```

**hidden_history方式の利点**：
- exit_fnが柔軟に定義可能（デフォルトはCALM式cos_sim）
- 各レイヤーの出力を分析可能
- ExitClassifierクラス不要でシンプル

---

## 設計原則

1. **事前学習専用** - generate、KVキャッシュは実装しない
2. **コンポジション方式** - LEGOBlockはTransformerBlockをラップ（継承ではない）
3. **exit_fn方式** - hidden_historyを受け取る関数でexit判定
4. **LEGOLLMはルーティングのみ** - Block間のインデックス管理と統計計算
5. **トークン単位のEarly Exit** - exit判定はトークン単位（バッチ単位ではない）
6. **TRUE Early Exit** - exitしたトークンの後続blockは処理しない
7. **訓練と推論の分離** - LEGOBlockは推論のみ、訓練は`train_block()`関数
8. **シーケンス単位処理** - Attention計算のためシーケンス全体を処理、exit判定のみトークン単位

---

## 核心機能（削除禁止）

1. `LEGOBlock.forward()` - Transformer処理 + exit判定（h, logits, should_exit, hidden_history）
2. `LEGOLLM.forward()` - TRUE Early Exit推論
3. `train_legollm()` - LEGOLLM全体の訓練（model_trainer.py）
4. `train_block()` - LEGOBlock訓練 + hard example収集（trainer.py）
5. `evaluate_legollm()` - LEGOLLM評価（model_trainer.py）
6. `SequenceData` - hidden states + targetsのコンテナ（シーケンス単位）

### TransformerBlockの責務

```python
class TransformerBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            - h_out: 最終出力
            - hidden_history: [input, layer1_out, layer2_out, ...]
        """
        hidden_history = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_history.append(x)
        return x, hidden_history
```

### LEGOBlockの責務

```python
# Type alias for exit function
ExitFn = Callable[[List[torch.Tensor], float], torch.Tensor]

def default_exit_fn(hidden_history: List[torch.Tensor], threshold: float) -> torch.Tensor:
    """CALM-style exit: cos_sim(hidden_history[-2], hidden_history[-1]) >= threshold"""
    h_in, h_out = hidden_history[-2], hidden_history[-1]
    cos_sim = F.cosine_similarity(h_in, h_out, dim=-1)
    return cos_sim >= threshold

class LEGOBlock(nn.Module):
    def __init__(self, transformer: TransformerBlock, exit_fn: Optional[ExitFn] = None):
        self.transformer = transformer
        self.exit_fn = exit_fn or default_exit_fn
        self.threshold = 0.0  # trainerが設定
        self.output_head: nn.Linear | None = None  # LEGOLLMが設定

    def forward(self, h) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        h_out, hidden_history = self.transformer(h)
        logits = self.output_head(h_out)
        should_exit = self.exit_fn(hidden_history, self.threshold)
        return h_out, logits, should_exit, hidden_history
```

### カスタムexit_fnの例

```python
# 全レイヤーの変化量を考慮するexit_fn
def multi_layer_exit_fn(hidden_history: List[torch.Tensor], threshold: float) -> torch.Tensor:
    """複数レイヤー間のcos_sim平均で判断"""
    cos_sims = []
    for i in range(1, len(hidden_history)):
        h_in, h_out = hidden_history[i-1], hidden_history[i]
        cos_sim = F.cosine_similarity(h_in, h_out, dim=-1)
        cos_sims.append(cos_sim)
    avg_cos_sim = torch.stack(cos_sims).mean(dim=0)
    return avg_cos_sim >= threshold

# 使用例
block = LEGOBlock(transformer, exit_fn=multi_layer_exit_fn)
```

---

## Hard Example収集方式（重要：削除禁止）

**トークン単位で収集**：`hard_ratio=0.5`ならcos_sim下位50%のトークンをhard exampleとして収集。

```python
# 正しい実装（トークン単位）- trainer.py の _collect_hard_examples()
# 1. hidden_historyからcos_simを計算
h_out, _, _, hidden_history = block.forward(h)
h_in = hidden_history[-2]
cos_sim = compute_cos_sim(h_in, h_out)

# 2. thresholdを計算（下位hard_ratio%がhard）
threshold = torch.quantile(all_cos_flat, hard_ratio)

# 3. 各トークンがhardかどうか判定
hard_token_mask = cos_sim < threshold  # (num_sequences, seq_len)

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
| **トークン単位収集** | 個々のトークン | 各トークンのcos_sim < threshold | hardトークンのみ抽出、repackして新シーケンス作成 |
| **シーケンス単位収集** | シーケンス全体 | シーケンス内の最小cos_simで判定 | hardシーケンス全体を選択（easyトークンも含む） |

### 禁止理由

1. **easyトークンがBlock 1に流れる**: シーケンス内の簡単なトークンも一緒に訓練される
2. **訓練効率の低下**: Block 0で十分処理できるトークンを再度訓練する無駄
3. **LEGOの設計思想に反する**: 「難しいものだけを深い層で処理」が核心

### 永続的な方針

**トークン単位収集のみを使用する**。シーケンス単位収集は今後一切実装しない。

---

## シーケンス単位処理の重要性（重要：削除禁止）

**Attentionはシーケンス全体を必要とする**：

```python
# 正しい実装（シーケンス単位）
h_out, logits, should_exit, hidden_history = block.forward(sequences)
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

### 4. モデルに訓練ロジックを含める

**問題：** LEGOBlockに`fit()`メソッドを実装し、300行超のクラスになっていた。

**教訓：** 訓練と推論を分離する。モデルは推論のみ、訓練は外部関数で。

### 5. トークン単位でAttention処理

**問題：** 訓練時にトークンを独立して `(batch, 1, dim)` で処理し、Attentionが機能していなかった。

**教訓：** Attentionはシーケンス全体を必要とする。シーケンス単位で処理し、exit判定のみトークン単位にする。

### 6. Hard example収集でシーケンス全体を渡す

**問題：** 「hardトークンを含むシーケンス全体」をBlock 1に渡し、easyトークンも含めて訓練していた。

**教訓：** Block 1が受け取るべきはhardトークンのみ。easyトークンのhidden statesはBlock 1に流れてはいけない。

### 7. 複数の信頼度計算方式を保持

**問題：** softmax方式、MLP方式、CALM方式など複数の方式をオプションとして保持していた。

**教訓：** 最良と判明した方式に一本化する。メンテナンス性 > 柔軟性。

### 8. 訓練データでvalidation PPLを計算

**問題：** `train_block()`内で訓練データを80/20分割し、20%側でval_pplを計算していた。

**教訓：** 訓練データとvalidationデータは完全に分離する。内部分割は混乱の元。

### 9. ExitClassifierクラスの肥大化

**問題：** ExitClassifierクラスにMLP、CALM、複数方式を混在させていた。

**解決（2024-12-15）：** exit_fn関数方式に変更。シンプルな関数で判定、hidden_historyを活用。

**教訓：** クラスよりも関数の方がシンプルで柔軟な場合がある。

---

## 頻発する設計ミス（⚠️ 要注意）

### 1. シーケンス単位処理とトークン単位early exitの混同

**原則**：
- **Attention計算** → シーケンス単位（`(batch, seq_len, dim)`）
- **Early exit判定** → トークン単位（各トークンが独立してexit）
- **Hard example収集** → トークン単位（hardトークンのみ抽出）
- **訓練時のforward** → シーケンス単位（Attentionのため）

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
