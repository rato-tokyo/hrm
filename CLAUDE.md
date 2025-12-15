# CASCADE (Confidence-Aware Sequential Compute Allocation for Dynamic Exit)

## CASCADEとは

**CASCADEは、複数のLLMを統合し、Early Exitで効率的にルーティングするフレームワークです。**

核心的な考え方：
- **任意のLLMを`LLM`クラスでラップ**
- 各LLMに**Early Exit機能を追加**し、簡単なトークンはそこで処理完了
- **Hard token（難しいトークン）だけ**を後段に渡す
- 滝（CASCADE）のように、処理が段階的に流れる

```
┌─────────────────────────────────────────────────────────┐
│                      Ensemble                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │  LLM 0  │───▶│  LLM 1  │───▶│  LLM 2  │ ───▶ ...   │
│  │+ Exit   │    │+ Exit   │    │+ Exit   │             │
│  └─────────┘    └─────────┘    └─────────┘             │
│      │              │              │                    │
│   exit here      exit here      exit here              │
│   (easy)         (medium)        (hard)                │
└─────────────────────────────────────────────────────────┘
```

## スコープ

**本フレームワークは事前学習（pre-training）専用です。**

- テキスト生成（generate）機能は含まない
- KVキャッシュは実装しない（事前学習では不要）

---

## 命名規則（重要）

### 方針：汎用的なクラス名を使用

**フレームワーク名が変更されても修正箇所を最小限にするため、クラス名には汎用的な名前を使用します。**

| クラス名 | 説明 | 理由 |
|----------|------|------|
| `LLM` | BaseLLMをラップするクラス | 「CascadeLLM」等にしない |
| `Ensemble` | 複数のLLMを統合するクラス | 「CascadeEnsemble」等にしない |
| `TransformerBlock` | Transformerの実装 | そのまま |

**禁止事項**:
- ❌ クラス名にフレームワーク名を含めない（例: `CascadeLLM`, `LEGOBlock`）
- ❌ フレームワーク固有の略称をクラス名に使わない

**理由**:
- フレームワーク名変更時の修正箇所を最小化
- インポート文やユーザーコードへの影響を抑制
- コードの可読性と汎用性を維持

### ファイル構成

```
cascade/
├── modules/
│   ├── transformer.py  # TransformerLayer, TransformerBlock
│   ├── attention.py    # MultiHeadAttention
│   ├── ffn.py          # GatedLinearUnit
│   └── norm.py         # RMSNorm
├── llm.py              # LLM（BaseLLM + Early Exit）
├── ensemble.py         # Ensemble（統合・評価）
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

## アーキテクチャ

```
BaseLLM     → 任意のLLM（TransformerBlockなど）
LLM         → BaseLLM + Early Exit機能（exit_fn + threshold + output_head）
Ensemble    → LLM × N の統合・ルーティング
```

### クラスの責務

| クラス | 責務 |
|--------|------|
| `BaseLLM` | 純粋なLLM処理（hidden states → hidden states） |
| `LLM` | BaseLLMをラップし、Early Exit機能を追加 |
| `Ensemble` | 複数のLLMを統合、ルーティング管理 |

---

## コア概念

### 統合フロー

1. **LLMをラップ**: 任意のBaseLLMを`LLM`クラスでラップ
2. **Ensembleで統合**: 複数の`LLM`を`Ensemble`に登録
3. **順次訓練**: 各LLMを訓練し、hard tokensを次に渡す
4. **Early Exit機能**: threshold設定、hard token判定
5. **推論時**: 簡単なトークンは前段でexit、難しいトークンだけ後段へ

### 重要な設計思想

**LLMクラスは汎用ラッパー**:
- 既存の訓練済みLLMをラップ可能
- 新規の未学習LLMもラップ可能
- Early Exit機能は自動的に追加される

**Hard tokenの定義**:
- cos_sim(入力hidden, 出力hidden)が低いトークン
- = LLMを通過することで大きく変化したトークン
- = そのLLMにとって「難しい」トークン

**Ensembleは動的に拡張可能**:
- `add_llm()`で新しいLLMを追加
- 追加されたLLMには自動的に共有output_headが設定される

---

## 設計原則

1. **汎用クラス名** - フレームワーク名をクラス名に含めない
2. **汎用ラッパー** - `LLM`クラスは任意のBaseLLMをラップ可能
3. **コンポジション方式** - `LLM`はBaseLLMをラップ（継承ではない）
4. **exit_fn方式** - hidden_historyを受け取る関数でexit判定
5. **Ensembleはルーティングのみ** - LLM間のインデックス管理と統計計算
6. **トークン単位のEarly Exit** - exit判定はトークン単位
7. **TRUE Early Exit** - exitしたトークンは後続LLMを実際に通過しない
8. **訓練と推論の分離** - モデルは推論のみ、訓練は外部関数で
9. **動的拡張** - `Ensemble.add_llm()`で後からLLMを追加可能

---

## 核心機能

### LLMクラスの責務

```python
class LLM(nn.Module):
    """
    BaseLLM + Early Exit機能を持つ汎用LLMラッパー。

    任意のLLMをラップし、以下を追加:
    - output_head: logits計算用（全LLMで共有）
    - exit_fn: exit判定関数
    - threshold: exit判定の閾値
    """

    def __init__(self, base_llm: nn.Module, exit_fn: Optional[ExitFn] = None):
        self.base_llm = base_llm
        self.exit_fn = exit_fn or default_exit_fn
        self.threshold = 0.0  # trainerが設定
        self.output_head: nn.Linear | None = None  # Ensembleが設定

    def train_forward(self, h) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """【訓練用】exit判定なし"""

    def evaluate(self, h) -> Tuple[Tensor, Tensor, Tensor]:
        """【評価用】exit判定あり"""
```

### Ensembleクラスの責務

```python
class Ensemble(nn.Module):
    """
    複数のLLMを統合。

    - Embedding層を保持
    - 共有output_headを管理
    - TRUE Early Exitによるルーティング
    - add_llm()で動的にLLMを追加
    """

    def __init__(self, vocab_size: int, dim: int, llms: List[LLM]):
        self.embedding = nn.Embedding(vocab_size, dim)
        self.llms = nn.ModuleList(llms)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # 全LLMでoutput_headを共有
        for llm in self.llms:
            llm.set_output_head(self.output_head)

    def add_llm(self, llm: LLM) -> None:
        """新しいLLMをアンサンブルに追加。"""
        llm.set_output_head(self.output_head)
        self.llms.append(llm)
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
   └─ 訓練完了

2. LLM 0 の hard tokens 収集
   ├─ 全データを通す
   ├─ cos_sim < threshold のトークンを抽出
   └─ threshold を LLM 0 に設定（推論時のexit判定用）

3. LLM 1 訓練（hard tokensのみ）
   └─ LLM 0 が苦手なトークンを専門に学習

4. 推論時
   ├─ 全トークンを LLM 0 に入力
   ├─ cos_sim >= threshold → exit（LLM 0 の予測を使用）
   └─ cos_sim < threshold → LLM 1 へ（hard tokensのみ）
```

---

## 使用例

```python
from cascade import (
    Ensemble,
    LLM,
    TransformerBlock,
    train_ensemble,
    evaluate_ensemble,
)

# 任意のLLMをLLMクラスでラップ
llm_0 = LLM(TransformerBlock(dim=64, num_heads=4, num_layers=2, ...))
llm_1 = LLM(TransformerBlock(dim=64, num_heads=4, num_layers=2, ...))

# Ensembleで統合
ensemble = Ensemble(vocab_size, dim=64, llms=[llm_0, llm_1])

# 訓練（各LLMは順番に訓練され、hard tokensを次に渡す）
train_ensemble(ensemble, train_data, val_data, config)

# 評価（TRUE Early Exit）
stats = evaluate_ensemble(ensemble, val_batches)
print(f"Shallow ratio: {stats['shallow_ratio']}")  # 前段LLMでexitした割合

# 後からLLMを追加することも可能
llm_2 = LLM(TransformerBlock(...))
ensemble.add_llm(llm_2)
```

---

## 過去の設計ミスと教訓

### 1. フレームワーク名をクラス名に含めた

**問題：** `LEGOBlock`, `LEGOEnsemble`など、フレームワーク名をクラス名に含めた。

**解決：** 汎用的な`LLM`, `Ensemble`に統一。名前変更時の修正箇所を最小化。

### 2. 「Block」という名前の誤解

**問題：** LEGOBlockをTransformerの「ブロック」と誤解。実際は独立したLLM。

**解決：** クラス名を`LLM`に変更し、汎用ラッパーであることを明確化。

### 3. TRUE Early Exitの誤実装

**問題：** 全トークンを全LLMに通してからマスクで統計を取る実装。

**教訓：** TRUE early exitなら実際に計算をスキップする。

### 4. トークン単位でAttention処理

**問題：** トークンを独立して処理し、Attentionが機能していなかった。

**教訓：** Attentionはシーケンス全体を必要とする。

### 5. Hard token収集でシーケンス全体を渡す

**問題：** hardトークンを含むシーケンス全体を後段LLMに渡していた。

**教訓：** 後段LLMにはhardトークンのみを渡す。

---

## ⛔ 禁止事項

1. **クラス名にフレームワーク名を含める** - 汎用名を使用
2. **シーケンス単位のhard収集** - トークン単位のみ使用
3. **easyトークンを後段LLMに渡す** - hardトークンのみ
4. **generateやKVキャッシュの実装** - 事前学習専用
5. **モデルクラスに訓練ロジック** - 外部関数で分離
