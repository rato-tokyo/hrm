# CASCADE (Confidence-Aware Sequential Compute Allocation for Dynamic Exit)

## CASCADEとは

**CASCADEは、複数のLLMを統合し、Early Exitで効率的にルーティングするフレームワークです。**

核心的な考え方：
- **任意のHugging Face CausalLMを`LLM`クラスでラップ**
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
| `LLM` | Hugging Face CausalLMをラップするクラス | 「CascadeLLM」等にしない |
| `Ensemble` | 複数のLLMを統合するクラス | 「CascadeEnsemble」等にしない |

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
├── __init__.py         # パッケージエクスポート
├── llm.py              # LLM（Hugging Face CausalLM + Early Exit）
├── ensemble.py         # Ensemble（統合・ルーティング）
├── exit_fn.py          # ExitFn, default_exit_fn, compute_cos_sim
├── llm_trainer.py      # train_llm()（単一LLM訓練）
├── llm_evaluator.py    # compute_ppl(), evaluate_llm()（単一LLM評価）
├── ensemble_trainer.py # train_ensemble(), create_sequence_data()
├── sequence_data.py    # SequenceData
├── config.py           # TrainerConfig, ExperimentConfig
├── dataloader.py       # create_wikitext_dataloaders
└── utils.py            # set_seed, get_device
```

---

## アーキテクチャ

```
CausalLM   → Hugging Face LLM（GPT2LMHeadModel, LlamaForCausalLM等）
LLM        → CausalLM + Early Exit機能（exit_fn + threshold）
Ensemble   → LLM × N のルーティング管理のみ
```

### クラスの責務

| クラス | 責務 |
|--------|------|
| `CausalLM` | Hugging Faceの完全なLLM（embedding + transformer + lm_head） |
| `LLM` | CausalLMをラップし、Early Exit機能を追加 |
| `Ensemble` | 複数のLLMを統合、ルーティング管理のみ |

---

## コア概念

### 統合フロー

1. **LLMをラップ**: 任意のHugging Face CausalLMを`LLM`クラスでラップ
2. **Ensembleで統合**: 複数の`LLM`を`Ensemble`に登録
3. **順次訓練**: 各LLMを訓練し、hard tokensを次に渡す
4. **Early Exit機能**: threshold設定、hard token判定
5. **推論時**: 簡単なトークンは前段でexit、難しいトークンだけ後段へ

### 重要な設計思想

**LLMクラスは自己完結した独立モデル**:
- 既存の訓練済みHugging Face CausalLMをそのままラップ
- embedding, transformer, lm_headは全て元モデルのものを使用
- Early Exit機能のみを追加
- 独自実装は不要（Hugging Faceに委譲）

**CausalLMを使用する理由**:
- 各LLMは独立したモデルとして自己完結すべき
- embedding, lm_headを独自管理する必要がない
- 訓練済みモデル（gpt2, llama等）をそのまま使用可能
- コードがシンプルになる

**Hard tokenの定義**:
- cos_sim(入力hidden, 出力hidden)が低いトークン
- = LLMを通過することで大きく変化したトークン
- = そのLLMにとって「難しい」トークン

**Ensembleはルーティングのみ**:
- embeddingを持たない（各LLMが持つ）
- `add_llm()`で新しいLLMを追加
- vocab_sizeとdimの一致を検証

---

## 設計原則

1. **汎用クラス名** - フレームワーク名をクラス名に含めない
2. **Hugging Face CausalLM互換** - `LLM`クラスは任意のCausalLMをラップ
3. **コンポジション方式** - `LLM`はCausalLMをラップ（継承ではない）
4. **LLMの自己完結** - 各LLMがembedding, transformer, lm_headを保持
5. **exit_fn方式** - hidden_historyを受け取る関数でexit判定
6. **Ensembleはルーティングのみ** - embeddingを持たない
7. **トークン単位のEarly Exit** - exit判定はトークン単位
8. **TRUE Early Exit** - exitしたトークンは後続LLMを実際に通過しない
9. **動的拡張** - `Ensemble.add_llm()`で後からLLMを追加可能

---

## 核心機能

### LLMクラスの責務

```python
from transformers import PreTrainedModel

class LLM(nn.Module):
    """
    Hugging Face CausalLM + Early Exit機能を持つ汎用LLMラッパー。

    任意のHugging Face CausalLMをラップし、Early Exit機能のみを追加。
    embedding, lm_headは元モデルのものをそのまま使用。
    """

    def __init__(self, base_llm: PreTrainedModel, exit_fn: Optional[ExitFn] = None):
        self.base_llm = base_llm
        self.exit_fn = exit_fn or default_exit_fn
        self.threshold = 0.0  # collect_hard_tokensで設定

    def forward(self, x, input_type="token_ids") -> Tuple[Tensor, List[Tensor]]:
        """token_idsまたはhidden_statesを処理"""

    def get_logits(self, h) -> Tensor:
        """base_llm.lm_headを使用してlogits計算"""
        return self.base_llm.lm_head(h)

    def collect_hard_tokens(self, data, hard_ratio, batch_size) -> SequenceData:
        """hard tokens収集、thresholdも設定"""

    def transform_data(self, data, batch_size) -> SequenceData:
        """データをこのLLMで変換"""
```

### LLM評価関数（llm_evaluator.py）

```python
def compute_ppl(llm, data, batch_size) -> float:
    """パープレキシティを計算（exit判定なし）"""

def evaluate_llm(llm, data, batch_size, is_last) -> Tuple[SequenceData, Dict]:
    """exit判定あり、統計と継続データを返す"""
```

### Ensembleクラスの責務

```python
class Ensemble(nn.Module):
    """
    複数のLLMを統合（ルーティングのみ）。

    - embeddingを持たない（各LLMが保持）
    - TRUE Early Exitによるルーティング
    - add_llm()で動的にLLMを追加
    """

    def __init__(self, llms: List[LLM]):
        self.llms = nn.ModuleList(llms)
        # vocab_sizeとdimの一貫性を検証

    def add_llm(self, llm: LLM) -> None:
        """新しいLLMをアンサンブルに追加（vocab_size, dim検証あり）。"""

    def evaluate(self, val_batches, batch_size) -> Dict:
        """【評価用】TRUE Early Exitで評価し、統計を返す"""
```

---

## Hard Token収集（重要）

**トークン単位で収集**：cos_sim下位N%のトークンがhard。

```python
# LLM.collect_hard_tokens()の内部処理
h_out, hidden_history = llm.forward(h, input_type="hidden_states")
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
from transformers import AutoModelForCausalLM, GPT2Config
from cascade import (
    Ensemble,
    LLM,
    train_ensemble,
    create_sequence_data,
    TrainerConfig,
)

# Hugging Face CausalLMを使用してLLMを作成
gpt2_config = GPT2Config(vocab_size=50257, n_embd=64, n_head=4, n_layer=2)
llm_0 = LLM(AutoModelForCausalLM.from_config(gpt2_config))
llm_1 = LLM(AutoModelForCausalLM.from_config(gpt2_config))

# または訓練済みモデルを使用
# llm_0 = LLM(AutoModelForCausalLM.from_pretrained("gpt2"))

# Ensembleで統合（vocab_size, dim引数不要）
ensemble = Ensemble([llm_0, llm_1])

# トークンを埋め込んでSequenceDataを作成
train_data = create_sequence_data(ensemble, train_batches)
val_data = create_sequence_data(ensemble, val_batches)

# 訓練（各LLMは順番に訓練され、hard tokensを次に渡す）
config = TrainerConfig(batch_size=32, max_epochs=50, ...)
train_ensemble(ensemble, train_data, val_data, config, lr_decay=0.5)

# 評価（TRUE Early Exit）
stats = ensemble.evaluate(val_batches, batch_size=32)
print(f"PPL: {stats['ppl']:.2f}, Accuracy: {stats['accuracy']:.2%}")

# 後からLLMを追加することも可能
llm_2 = LLM(AutoModelForCausalLM.from_config(gpt2_config))
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

### 6. 独自Transformer実装のメンテナンス負担

**問題：** RMSNorm, RoPE, GLUなど独自実装し、約200行のコードを維持。

**解決：** Hugging Face Transformersに移行。メンテナンス負担を大幅削減。

### 7. output_headの独自管理

**問題：** 各LLMにoutput_headを独自実装・管理していた。

**解決：** CausalLM（lm_head内蔵）を使用。独自管理不要に。

---

## ⛔ 禁止事項

1. **クラス名にフレームワーク名を含める** - 汎用名を使用
2. **シーケンス単位のhard収集** - トークン単位のみ使用
3. **easyトークンを後段LLMに渡す** - hardトークンのみ
4. **generateやKVキャッシュの実装** - 事前学習専用
5. **モデルクラスに訓練ロジック** - 外部関数で分離
6. **独自Transformer実装** - Hugging Face Transformersを使用
7. **output_headの独自管理** - CausalLMのlm_headを使用
8. **Ensembleにembeddingを持たせる** - 各LLMが保持
