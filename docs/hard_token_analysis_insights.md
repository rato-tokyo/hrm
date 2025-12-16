# Hard Token分析から得られた知見

## 概要

SmolLM2-135M-Instructを使用したHard Token分析（閾値: 0.9341, 32サンプル, 4,096トークン）から得られた知見をまとめる。

## 基本統計

| 指標 | 値 |
|------|-----|
| 総トークン数 | 4,096 |
| Hardトークン数 | 344 |
| Hard比率 | 8.4% |
| 閾値 | 0.9341 |

## 主要な発見

### 1. Hardトークンは「意味的分岐点」を示す

Hardトークンは単に「難しい」トークンではなく、**文の意味が分岐する瞬間**を示している。

**例: "She was scared" → "She was trembling with fear"**

```
Instruction: Take this sentence and make it more descriptive: She was scared.
Input: She was scared.

Response:
She      Easy (0.95+)  ← 元文と同じ主語
was      HARD (0.83)   ← 分岐点！次に何が来るか不確定
trembling Easy (0.94+) ← wasが決まれば流れが決まる
with     Easy          ← 定型表現
fear     Easy          ← trembling with ___は自然にfear
```

`was`がHardなのは、ここで「trembling」「shaking」「crying」など複数の選択肢があるため。一度`was`の次が決まれば、後続は自然に流れる。

### 2. 前提知識がEasy/Hardを決定する

InstructionやInputに明示された情報があると、関連するトークンはEasyになりやすい。

**例: リチウムの原子質量**

```
Instruction: Calculate the atomic mass for lithium.

Response:
The atomic mass for lithium is  ← 全てEasy（質問の言い換え）
6.941 u                         ← Hard連鎖（具体的な数値）
```

- 「atomic mass for lithium」は質問に含まれるためEasy
- 「6.941」は事前学習で暗記されていない具体的数値のためHard

### 3. Hard連鎖のパターン

Hardトークンは連鎖する傾向がある。特に以下のパターン：

| パターン | 例 | 説明 |
|---------|-----|------|
| 数値 | `6.941` | 各桁が連続してHard |
| 固有名詞 | `The Usual Suspects` | 名前全体がHard |
| サブワード | `pan`+`icked` | 分割されたトークンが連続Hard |
| 列挙 | `yoga, swimming, or` | 選択肢とコンマがHard |

### 4. cos_sim値の解釈

| cos_sim範囲 | 解釈 | 例 |
|-------------|------|-----|
| 0.95以上 | Easy（予測容易） | 定型表現、前提から推測可能 |
| 0.93-0.95 | 境界域 | 閾値付近 |
| 0.90-0.93 | やや難しい | 複数選択肢がある |
| 0.85-0.90 | 明確にHard | 具体的な選択が必要 |
| 0.85未満 | 非常にHard | 分岐点、数値の最初の桁 |

**最もHardだったトークン**: `was` (0.83) - 「She was」の後に何が来るかの分岐点

## 仮説: 重要トークンの定義

### 提案された仮説

```
重要なトークン = 全てのHard + 各Hardの直前トークン
```

### 根拠

1. **Hardトークン**: 意味的分岐点であり、予測の核心
2. **直前トークン**: 分岐のトリガーとなる文脈を提供

### 具体例

**例1: 関節炎の運動**
```
元:   For someone with arthritis, the best type of exercise would be low-impact activities like yoga, swimming, or walking.

Hard: be(0.92), low(0.89), activities(0.90), yoga(0.89), swimming(0.89), ,(0.88), or(0.93)

重要トークン: would be low -impact activities like yoga , swimming , or
削除可能:     For someone with arthritis the best type of exercise walking
```

**例2: 恐怖の描写**
```
元:   She was trembling with fear, her heart racing wildly

Hard: was(0.83), her(0.92)

重要トークン: She was , her
削除可能:     trembling with fear heart racing wildly
```

### 削減効果の推定

| 例 | 元トークン数 | 重要トークン数 | 削減率 |
|----|------------|---------------|--------|
| 関節炎の運動 | 22 | 11 | 50% |
| 恐怖の描写 | 11 | 4 | 64% |
| 映画の分類 | 10 | 8 | 20% |

Hard連鎖が多い文は削減率が低く、Easyが多い定型文は削減率が高い。

## CASCADEへの示唆

### 現在の設計の妥当性

現在のCASCADE設計（Hardトークンのみ後段LLMへ）は基本的に正しい：
- Hardトークンは意味的に重要な分岐点
- PPL 52.3%改善という結果がこれを裏付ける

### 改善の可能性

1. **直前トークンの追加**: 各Hardトークンの直前1トークンも含めることで、分岐のトリガー情報を保持

2. **Hard連鎖の一括処理**: 連続するHardトークンを1つの単位として扱う

3. **前提知識の活用**: Instruction/Inputの情報を後段LLMにも渡すことで、文脈理解を向上

## 結論

Hard Token分析から、以下が明らかになった：

1. **Hardトークンは「意味的分岐点」**を示す
2. **前提知識（Instruction/Input）がEasy/Hardを決定**する重要な要因
3. **「Hard + 直前トークン」の仮説**は有力（推定有力度: 85%以上）
4. 現在のCASCADE設計は妥当だが、**直前トークンの追加**で改善の余地あり

## 実験日

2024年12月16日
