# LEGO (Layered Ensemble with Gradual Optimization) の仕様

## LEGOとは

**LEGOは、既存のLLMに新しいLLMを段階的に追加・統合するフレームワークです。**

核心的な考え方：
- **LLM 0（前段）は既存の訓練済みLLM**
- 前段LLMに**Early Exit機能を追加**し、簡単なトークンはそこで処理完了
- **Hard token（難しいトークン）だけ**を後段に渡す
- **LLM 1以降は未学習のLLM**で、hard tokensのみで訓練される
- これにより、既存LLMを拡張できる

```
┌─────────────────┐     hard tokens     ┌─────────────────┐
│  既存LLM (訓練済)│ ─────────────────── │  新LLM (未学習)  │
│ (+ Early Exit)  │                     │ hard tokensで訓練│
└─────────────────┘                     └─────────────────┘
        │                                       │
   easy tokens                             hard tokens
   ここでexit                               ここでexit
```

## スコープ

**本フレームワークは事前学習（pre-training）専用です。**

- テキスト生成（generate）機能は含まない
- KVキャッシュは実装しない（事前学習では不要）

---

## アーキテクチャ

```
BaseLLM             → 任意のLLM（TransformerBlockなど）
EarlyExitLLM        → BaseLLM + Early Exit機能（exit_fn + threshold + output_head）
LEGOEnsemble        → EarlyExitLLM × N の統合・ルーティング
```

### クラスの責務

| クラス | 責務 |
|--------|------|
| `BaseLLM` | 純粋なLLM処理（hidden states → hidden states） |
| `EarlyExitLLM` | BaseLLMをラップし、Early Exit機能を追加 |
| `LEGOEnsemble` | 複数のEarlyExitLLMを統合、ルーティング管理 |

### ファイル構成

```
lego/
├── modules/
│   ├── transformer.py  # TransformerLayer, TransformerBlock（BaseLLMの一例）
│   ├── attention.py    # MultiHeadAttention
│   ├── ffn.py          # GatedLinearUnit
│   └── norm.py         # RMSNorm
├── early_exit_llm.py   # EarlyExitLLM（BaseLLM + Early Exit）
├── lego_ensemble.py    # LEGOEnsemble（統合・評価）
├── exit_fn.py          # ExitFn, default_exit_fn, compute_cos_sim
├── llm_trainer.py      # train_llm()（単一LLM訓練）
├── ensemble_trainer.py # train_ensemble()（全体訓練）
├── evaluator.py        # evaluate_ensemble()
├── sequence_data.py    # SequenceData
├── config.py           # TrainerConfig, ExperimentConfig
├── dataloader.py       # create_wikitext_dataloaders
└── utils.py            # set_seed, get_device
```

---

## コア概念

### 統合フロー

1. **LLM 0（既存LLM）を訓練**: 全データで言語モデルとして訓練
2. **Early Exit機能を追加**: threshold設定、hard token判定
3. **Hard tokensを収集**: LLM 0で処理しきれなかったトークン
4. **LLM 1（新規・未学習）を訓練**: Hard tokensのみで訓練
5. **推論時**: 簡単なトークンはLLM 0でexit、難しいトークンだけLLM 1へ

### 重要な設計思想

**LLM 0は既存の訓練済みLLM**:
- 単体で完全に機能する言語モデル
- Early Exit機能は「後付け」で追加される

**LLM 1以降は未学習のLLM**:
- 初期状態では何も学習していない
- **Hard tokensのみ**で訓練される
- 前段LLMが苦手なトークンを専門に処理

**Hard tokenの定義**:
- cos_sim(入力hidden, 出力hidden)が低いトークン
- = LLMを通過することで大きく変化したトークン
- = そのLLMにとって「難しい」トークン

---

## 設計原則

1. **既存LLMの拡張** - LLM 0は訓練済み、LLM 1+は未学習でhard tokensを学習
2. **コンポジション方式** - EarlyExitLLMはBaseLLMをラップ（継承ではない）
3. **exit_fn方式** - hidden_historyを受け取る関数でexit判定
4. **LEGOEnsembleはルーティングのみ** - LLM間のインデックス管理と統計計算
5. **トークン単位のEarly Exit** - exit判定はトークン単位
6. **TRUE Early Exit** - exitしたトークンは後続LLMを実際に通過しない
7. **訓練と推論の分離** - モデルは推論のみ、訓練は外部関数で

---

## 核心機能

### EarlyExitLLMの責務

```python
class EarlyExitLLM(nn.Module):
    """
    BaseLLM + Early Exit機能。

    任意のLLMをラップし、以下を追加:
    - output_head: logits計算用（全LLMで共有）
    - exit_fn: exit判定関数
    - threshold: exit判定の閾値
    """

    def __init__(self, base_llm: nn.Module, exit_fn: Optional[ExitFn] = None):
        self.base_llm = base_llm
        self.exit_fn = exit_fn or default_exit_fn
        self.threshold = 0.0  # trainerが設定
        self.output_head: nn.Linear | None = None  # LEGOEnsembleが設定

    def train_forward(self, h) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """【訓練用】exit判定なし"""
        h_out, hidden_history = self.base_llm(h)
        logits = self.output_head(h_out)
        return h_out, logits, hidden_history

    def evaluate(self, h) -> Tuple[Tensor, Tensor, Tensor]:
        """【評価用】exit判定あり"""
        h_out, hidden_history = self.base_llm(h)
        logits = self.output_head(h_out)
        should_exit = self.exit_fn(hidden_history, self.threshold)
        return h_out, logits, should_exit
```

### LEGOEnsembleの責務

```python
class LEGOEnsemble(nn.Module):
    """
    複数のEarlyExitLLMを統合。

    - Embedding層を保持
    - 共有output_headを管理
    - TRUE Early Exitによるルーティング
    """

    def __init__(self, vocab_size: int, dim: int, llms: List[EarlyExitLLM]):
        self.embedding = nn.Embedding(vocab_size, dim)
        self.llms = nn.ModuleList(llms)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # 全LLMでoutput_headを共有
        for llm in self.llms:
            llm.set_output_head(self.output_head)
```

---

## Hard Token収集（重要）

**トークン単位で収集**：cos_sim下位N%のトークンがhard。

```python
# LLMを通過後のcos_simを計算
h_out, _, hidden_history = llm.train_forward(h)
h_in = hidden_history[-2]
cos_sim = compute_cos_sim(h_in, h_out)

# threshold以下がhard（大きく変化した = 難しい）
hard_mask = cos_sim < threshold
hard_hidden = h_out[hard_mask]

# 新シーケンスに再構成して次のLLMへ
```

**重要**: 後段LLMが受け取るのはhardトークンのみ。easyトークンは前段LLMで処理完了。

---

## 訓練フロー詳細

```
【全体の流れ】

1. LLM 0 訓練（全データ）
   └─ 訓練済みLLMとして動作可能に

2. LLM 0 の hard tokens 収集
   ├─ 全データを通す
   ├─ cos_sim < threshold のトークンを抽出
   └─ threshold を LLM 0 に設定（推論時のexit判定用）

3. LLM 1 訓練（hard tokensのみ）
   ├─ 未学習状態からスタート
   └─ LLM 0 が苦手なトークンを専門に学習

4. 推論時
   ├─ 全トークンを LLM 0 に入力
   ├─ cos_sim >= threshold → exit（LLM 0 の予測を使用）
   └─ cos_sim < threshold → LLM 1 へ（hard tokensのみ）
```

---

## 使用例

```python
from lego import (
    LEGOEnsemble,
    EarlyExitLLM,
    TransformerBlock,
    train_ensemble,
    evaluate_ensemble,
)

# 既存LLM（LLM 0）と新規LLM（LLM 1、未学習）を作成
llm_0 = EarlyExitLLM(TransformerBlock(dim=64, num_heads=4, num_layers=2, ...))
llm_1 = EarlyExitLLM(TransformerBlock(dim=64, num_heads=4, num_layers=2, ...))

# LEGOで統合
ensemble = LEGOEnsemble(vocab_size, dim=64, llms=[llm_0, llm_1])

# 訓練
# - LLM 0: 全データで訓練
# - LLM 1: LLM 0 の hard tokens で訓練
train_ensemble(ensemble, train_data, val_data, config)

# 評価（TRUE Early Exit）
stats = evaluate_ensemble(ensemble, val_batches)
print(f"Shallow ratio: {stats['shallow_ratio']}")  # LLM 0でexitした割合
```

---

## 過去の設計ミスと教訓

### 1. 「Block」という名前の誤解

**問題：** LEGOBlockをTransformerの「ブロック」と誤解。実際は独立したLLM。

**解決：** クラス名をEarlyExitLLMに変更し、概念を明確化。

### 2. TRUE Early Exitの誤実装

**問題：** 全トークンを全LLMに通してからマスクで統計を取る実装。

**教訓：** TRUE early exitなら実際に計算をスキップする。

### 3. トークン単位でAttention処理

**問題：** トークンを独立して処理し、Attentionが機能していなかった。

**教訓：** Attentionはシーケンス全体を必要とする。

### 4. Hard token収集でシーケンス全体を渡す

**問題：** hardトークンを含むシーケンス全体を後段LLMに渡していた。

**教訓：** 後段LLMにはhardトークンのみを渡す。

---

## ⛔ 禁止事項

1. **シーケンス単位のhard収集** - トークン単位のみ使用
2. **easyトークンを後段LLMに渡す** - hardトークンのみ
3. **generateやKVキャッシュの実装** - 事前学習専用
4. **モデルクラスに訓練ロジック** - 外部関数で分離
