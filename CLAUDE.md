# HRM Project - Claude Instructions

## Project Overview

**LEGO: Layered Ensemble with Gradual Optimization**

レゴブロックのようにStage（層グループ）を組み合わせる柔軟な訓練アーキテクチャ。

**コア技術**:
- **Stage-based Training**: 層をStageというブロックに分割し、柔軟に組み合わせる
- **ASHEM (Adaptive Supervision via Hard Example Mining)**: Hard examplesに特化した2-Stage訓練戦略
- **Early Exit**: 推論時の計算効率化

**実装状況**:
- ✅ 2-Stage LEGO (ASHEM): 実装完成、動作確認済み ([docs/ASHEM_STAGE_SPECIFICATION.md](docs/ASHEM_STAGE_SPECIFICATION.md))
- 🔄 N-Stage LEGO: 概念提案済み、実装予定

---

## 🚨 重要な実装上の注意事項

### Per-token Filtering - ASHEM実装の必須仕様

**⚠️ CRITICAL**: ASHEMのHard Example Miningでは**Per-token filtering**を使用することが必須です。

**動作する実装**: コミット **fc9b140** (Consolidate LASH to 2 core options)
- `src/ease/ashem.py`: Per-token filtering実装
- `colab2.py`: 動作確認済み実験スクリプト

**Per-token filteringの実装**:
```python
def compute_confidence_threshold(model, val_batches, target_ratio, device):
    """Per-token quantile calculation"""
    all_confidences = []
    for x, _ in val_batches:
        h = model.forward_to_layer(x, model.num_layers)
        confidence = compute_confidence(model, h)
        all_confidences.append(confidence.view(-1))  # ← Flatten per-token

    all_confidences = torch.cat(all_confidences)
    threshold = torch.quantile(all_confidences, target_ratio).item()
    return threshold

def collect_hard_examples(model, val_batches, threshold, device):
    """Per-token filtering"""
    for x, y in val_batches:
        h = model.forward_to_layer(x, model.num_layers)
        confidence = compute_confidence(model, h)

        # Per-token comparison
        mask = confidence < threshold  # (batch, seq_len)

        x_flat = x.view(-1)
        h_flat = h.view(-1, h.shape[-1])
        y_flat = y.view(-1)
        mask_flat = mask.view(-1)

        hard_inputs.append(x_flat[mask_flat])
        hard_hidden_states.append(h_flat[mask_flat])
        hard_targets.append(y_flat[mask_flat])
```

**期待される実験結果** (WikiText-2, 10K samples):
- Stage 1 Hard PPL: ~2,763
- Stage 2 Hard PPL: ~668
- Hard PPL Improvement: 75.8%
- Collected hard examples: ~32,768 (50% of total tokens)

**禁止事項**:
- ❌ Sequence-level averaging (`.mean(dim=1)`) を使用すること
- ❌ Per-token thresholdとSequence-level averageを混在させること

**理由**: Threshold計算とフィルタリングの方法が一致していないと、hard examplesが正しく収集されず、実験が失敗します。

---

## LEGO アーキテクチャ概要

### コアコンセプト: レゴブロックのような組み合わせ

**LEGO**の2つのコアオプションで全てを制御：

| オプション | 説明 | Reference |
|-----------|------|-----------|
| **stages** | どのStageブロックで学習するか | - |
| **routing_threshold** | 推論時Early Exit閾値 | Teerapittayanon et al., 2016 |

**重要**: Standard, Deep Supervision, ASHEMは全てLEGOアーキテクチャの異なる組み合わせパターン。

### 設定例：柔軟な組み合わせ

#### パターン1: Standard Transformer（従来型LLM）
```python
from ease import TrainingConfig, StageConfig

config = TrainingConfig(
    stages=[
        StageConfig(layers=(3, 3), loss_weight=1.0)  # 最終層のみ（1 stage）
    ]
)
```

#### パターン2: Deep Supervision（全層均等）
```python
config = TrainingConfig(
    stages=[
        StageConfig(layers=(1, 1), loss_weight=0.33),  # Layer 1
        StageConfig(layers=(2, 2), loss_weight=0.33),  # Layer 2
        StageConfig(layers=(3, 3), loss_weight=0.33),  # Layer 3
    ]
)
```

#### パターン3: ASHEM（2-Stage訓練）
```python
# Stage 1: Layer 1-2, Stage 2: Layer 3-4
config = TrainingConfig(
    stages=[
        StageConfig(layers=(1, 2), loss_weight=1.0),  # Stage 1: 浅層
        StageConfig(layers=(3, 4), loss_weight=1.0),  # Stage 2: 深層
    ],
    routing_threshold=0.95,  # 推論時Early Exit
    exit_layer=2
)
```

#### パターン4: カスタム（非対称重み）
```python
# Layer 1-2に重点、Layer 3は軽め
config = TrainingConfig(
    stages=[
        StageConfig(layers=(1, 2), loss_weight=0.7),
        StageConfig(layers=(3, 3), loss_weight=0.3),
    ],
    routing_threshold=0.9,
    exit_layer=2
)
```

---

## ドキュメント管理ポリシー

### ファイル構成セクションの削除理由

**CLAUDE.mdにファイル構成セクションは記載しない**

**理由**:
- ファイル構成は頻繁に変更される（リファクタリング、新機能追加等）
- ドキュメントの更新忘れによる情報の陳腐化を防ぐ
- 実際のコードベースを見れば構成は把握できる
- Globツールで簡単に確認可能: `**/*.py`

**推奨アプローチ**:
- 重要なのは「使い方」と「概念」
- ファイルの場所はインポート例で十分
- 構造的な説明が必要な場合は、コメントやdocstringに記載

---

## コードモジュール構成

### LEGO フレームワーク (src/ease/)

**コアモジュール**:
- `models.py` - StandardTransformer, DeepSupervisionTransformer
- `trainer.py` - StageConfig, TrainingConfig, Trainer (Stage-based訓練フレームワーク)
- `ashem.py` - ASHEMConfig, ASHEM訓練戦略（Per-token filtering実装）
- `modules/` - TransformerBlock, Attention, FFN, RMSNorm等

**実験ユーティリティ (experiments/)**:
- `utils.py` - データローダー、デバイス管理、seed設定

**実験スクリプト (root)**:
- `colab2.py` - ASHEM実験メインスクリプト（fc9b140で動作確認済み）

---

## 使用方法

### 基本的な使用例

```python
import sys
sys.path.insert(0, 'src')

from ease import DeepSupervisionTransformer, Trainer, TrainingConfig, StageConfig

# モデル作成
model = DeepSupervisionTransformer(vocab_size=1000, dim=64, num_layers=3)

# 設定: LASHの2つのコアオプションで全てを制御
config = TrainingConfig(
    stages=[
        StageConfig(layers=(1, 2), loss_weight=0.7),  # Stage 1: Layer 1-2
        StageConfig(layers=(3, 3), loss_weight=0.3),  # Stage 2: Layer 3
    ],
    routing_threshold=0.95,  # Early Exit閾値
    exit_layer=2
)

# 訓練
trainer = Trainer(config, vocab_size=1000)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)
loss = trainer.train_epoch(model, train_batches, optimizer)

# 評価
stats = trainer.evaluate(model, val_batches)
```

### ヘルパー関数（便利な設定プリセット）

```python
from ease import create_standard_config, create_deep_supervision_config

# Standard LLM設定（最終層のみ）
config = create_standard_config(num_layers=3)
# → stages=[StageConfig(layers=(3, 3), loss_weight=1.0)]

# Deep Supervision設定（全層均等）
config = create_deep_supervision_config(num_layers=3)
# → stages=[StageConfig(layers=(1, 1), 0.33), StageConfig(layers=(2, 2), 0.33), StageConfig(layers=(3, 3), 0.33)]
```

**注意**: これらはあくまでプリセット。`TrainingConfig`で自由にカスタマイズ可能。

### Early Stopping（訓練時の早期終了）

```python
# Early Stopping付き訓練
result = trainer.train_with_early_stopping(
    model=model,
    train_batches=train_loader,
    val_batches=val_loader,
    optimizer=optimizer,
    max_epochs=100,
    patience=1,  # デフォルト値: 1エポック改善なしで停止
    verbose=True
)
```

**重要ルール**:
- **patienceのデフォルト値は1**
- 検証損失が1エポックでも悪化したら訓練を停止
- 過学習を防ぎ、訓練時間を短縮
- 最良モデルの状態を自動保存・復元

### Perplexity (PPL) の解釈

**正常な値の範囲**:
- **小規模データ（1K サンプル）**: PPL 100-3000 程度
- **中規模データ（10K サンプル）**: PPL 10-1000 程度
- **大規模データ**: PPL 2-100 程度

**計算式**: `PPL = exp(avg_loss)`
- vocab_size=1000のランダム予測: loss ≈ log(1000) ≈ 6.9, PPL ≈ 1000
- loss=7.3 → PPL ≈ 1500（小規模データでは正常）
- loss=2.3 → PPL ≈ 10（十分に学習済み）

**注意**: PPLは指数関数的に増加するため、lossがわずかに高いだけでPPLは大きく見えます。**Accuracyで評価**することを推奨。

---

## LEGO 訓練戦略

LEGOアーキテクチャは3つの訓練戦略をサポート：

### 1. Standard LEGO
最終層のみで学習（従来のLLM訓練）= **1つのStageブロック**
```python
config = TrainingConfig(stages=[
    StageConfig(layers=(3, 3), loss_weight=1.0)  # 最終層のみの1ブロック
])
```

### 2. Deep Supervision LEGO
全層で均等に学習 = **全層を個別Stageブロックとして定義**
```python
config = TrainingConfig(stages=[
    StageConfig(layers=(1, 1), loss_weight=0.33),  # ブロック1
    StageConfig(layers=(2, 2), loss_weight=0.33),  # ブロック2
    StageConfig(layers=(3, 3), loss_weight=0.33),  # ブロック3
])
```

### 3. ASHEM LEGO
Hard examplesに特化した**2-Stageブロック訓練戦略**

**新規性**: 両サーベイ論文（2024-2025）にEarly ExitとHard Example Miningの組み合わせに関する記述なし

**訓練手順**:
- **Stage 1 Block**: 浅層ブロック（Layer 1-2）で全データ訓練 → Hard example識別
- **Stage 2 Block**: 深層ブロック（Layer 3-4）でHard examplesのみ訓練
- **推論**: 2つのブロックを動的に切り替え（Early Exit）

**LEGOブロックの組み合わせ**:
```python
# ブロック1: Layer 1-2（浅層）
# ブロック2: Layer 3-4（深層、Hard examplesのみ）
config = TrainingConfig(
    stages=[
        StageConfig(layers=(1, 2), loss_weight=1.0),  # ブロック1
        StageConfig(layers=(3, 4), loss_weight=1.0),  # ブロック2
    ],
    routing_threshold=0.15,  # 推論時ブロック切り替え閾値
    exit_layer=2
)
```

**実験結果** (WikiText-2, 10K samples):
- Hard PPL: **75.8%改善** (2763 → 668)
- 計算コスト: **36%削減** (64.82% of full model)
- Overall PPL: **15.9%改善** (986 → 830)

**使用例**:
```python
from ease import ASHEMConfig

ashem_config = ASHEMConfig(
    phase1_layers=2,        # Stage 1の層数
    hard_example_ratio=0.5, # Hard example収集率
    phase2_layers=4,        # Stage 2の総層数
)
```

詳細: [docs/experiments/hard_example_mining.md](docs/experiments/hard_example_mining.md)

**ASHEM の詳細仕様**: [docs/ASHEM_STAGE_SPECIFICATION.md](docs/ASHEM_STAGE_SPECIFICATION.md)
- Stage (ステージ) の正確な定義
- Per-token Filtering の必須仕様
- Early Exit の必須使用
- 実験結果の検証方法

**注意**: SDS の実装は未完成。現在は ASHEM (2-Stage) のみ動作確認済み (commit fc9b140)。

---

## パフォーマンス最適化

### compute_loss() の自動最適化

**LASHの2つのオプションを完全に維持したまま、訓練速度を最適化**:

```python
# 最終層のみ（高速パス使用）
config = TrainingConfig(layer_weights={1: 0, 2: 0, 3: 1})
# → forward() を使用（約8%高速化）

# 複数層（汎用パス使用）
config = TrainingConfig(layer_weights={1: 0.33, 2: 0.33, 3: 0.33})
# → forward_all_layers() を使用

# 非対称（汎用パス使用）
config = TrainingConfig(layer_weights={1: 0.7, 2: 0, 3: 0.3})
# → forward_all_layers() を使用
```

**最適化の仕組み**:
- `layer_weights` を解析し、最終層のみ必要な場合を検出
- 最終層のみの場合 → `forward()` 使用（中間層でoutput_headを実行しない）
- それ以外 → `forward_all_layers()` 使用（従来通り）

**互換性保証**:
- ✅ `layer_weights`: すべてのパターンで動作
- ✅ `routing_threshold`: 独立（評価時のみ使用）

**実測効果**（WikiText-2, 10K samples）:
- 最終層のみ: **8.4%高速化**（25.51秒 → 23.38秒）
- 複数層: 変化なし（すでに最適）

---

## References

### LASH Framework
- **LASH**: Layered Adaptive Supervision Hierarchy（本フレームワーク）
- Lee et al. (2015) - Deep Supervision
- Howard & Ruder (2018) - Discriminative Fine-Tuning
- Teerapittayanon et al. (2016) - Early Exit (BranchyNet)

### ASHEM Training Strategy
- **ASHEM**: Adaptive Supervision via Hard Example Mining（本研究）
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- **注意**: "Progressive Layer Addition"ではなく"Selective Layer Expansion"を使用（PLD (NeurIPS 2020)との混同を避けるため）

### Early Exit Surveys (新規性検証用)
- **ACM Survey** (Nov 2024): "Early-Exit Deep Neural Network—A Comprehensive Survey" (37 pages)
  - Haseena Rahmath P et al., ACM Computing Surveys
  - DOI: 10.1145/3698767
- **NLP Survey** (Jan 2025): "A Survey of Early Exit Deep Neural Networks in NLP" (13 pages)
  - Divya Jyoti Bajpai and Manjesh Kumar Hanawal
  - arXiv:2501.07670v1

**重要な知見**:
- 両サーベイとも、Early Exitの文脈での層ごとの学習率制御（`layer_lr_scales`）に言及なし
- 両サーベイとも、Early ExitとHard Example Miningの組み合わせに言及なし
- 既存研究では`wi = i`（深い層ほど重みが大きい）が一般的

---

## コードアーキテクチャ

### モジュール分離原則

**訓練フレームワークと訓練戦略の分離**:
- `trainer.py` - コア訓練フレームワーク（TrainingConfig, Trainer）
- `ashem.py` - ASHEM訓練戦略専用モジュール（Per-token filtering実装）

**分離の利点**:
- 明確な責務分離: フレームワーク vs 戦略
- 拡張性: 新しい訓練戦略を独立したモジュールとして追加可能
- 保守性: 各モジュールが特定の責務に集中

**将来の拡張例**:
```python
# 新しい訓練戦略を追加する場合
src/ease/
├── trainer.py      # コアフレームワーク（変更不要）
├── ashem.py        # ASHEM戦略
└── new_strategy.py # 新しい戦略（独立したモジュール）
```

---

## 実験実行原則

### Google Colab実行を前提

**重要**: しばらくの間、すべての実験はGoogle Colabで実行します。

#### 理由
- GPU（NVIDIA L4等）の利用可能性
- 大規模データセット（WikiText-2等）の高速処理
- 長時間訓練の安定実行

#### 実行スクリプト
- **メイン実験**: `colab2.py` (ASHEM実験、fc9b140で動作確認済み)
- ローカル実行用スクリプトは削除済み

#### Colab実行時の注意点

**データローダー**:
```python
# datasets ライブラリのインストール必要
!pip install datasets

# 自動的にHugging Faceからダウンロード
from experiments.utils import create_wikitext_dataloaders
```

**GPU確認**:
```python
import torch
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**実行コマンド**:
```bash
# Colabセルで実行
!python colab2.py
```

#### Git操作（動作するバージョンへの切り替え）
```bash
# 動作確認済みのコミットに切り替え
git checkout fc9b140

# または、最新のmainブランチを使用（fc9b140と同じ）
git checkout main
```

---

## 論文投稿方針

### 投稿先

**確定**: arXiv（統一フレームワーク論文として投稿）

### 論文タイトル案

**メイン**: LASH: Layered Adaptive Supervision Hierarchy for Efficient Transformer Training

**サブタイトル**: A Unified Framework with 2 Core Options

### 新規性（Novelty）の主張

**参考文献**:
- ACM Survey (Nov 2024): "Early-Exit Deep Neural Network—A Comprehensive Survey" (37 pages)
- NLP Survey (Jan 2025): "A Survey of Early Exit Deep Neural Networks in NLP" (13 pages)

#### 1. 統一フレームワークとしての新規性

**既存研究の問題点**:
- Deep Supervision、Early Exitは個別に提案された
- これらを組み合わせるには別々の実装が必要
- 柔軟な戦略カスタマイズが困難

**LASHの貢献**:
- 2つのコアオプション（`layer_weights`, `routing_threshold`）で全ての戦略を統一的に実現
- 単一フレームワークで3つ以上の訓練戦略をサポート
- 無限の戦略カスタマイズが可能

**各オプションの新規性分析**:

1. **`layer_weights` (層ごとの損失重み)**:
   - ⚠️ 既存研究で使用されている（両サーベイ論文で確認）
   - 最も一般的なパターンは `wi = i`（深い層ほど重みが大きい）
   - ✅ **LASHの独自性**: 任意の非対称パターンが可能（例: `{1: 0.7, 2: 0, 3: 0.3}`）
   - ✅ ゼロ重みによる層のスキップが可能

2. **`routing_threshold` (Early Exit閾値)**:
   - ⚠️ Early Exit自体は既存技術（Teerapittayanon et al., 2016 - BranchyNet以降）
   - ✅ **LASHの独自性**: layer_weightsとの統合による柔軟な制御

**修正されたClaim**:
"While existing work uses layer-wise loss weights with simple patterns (typically wi=i) [Survey'24], LASH is the first framework to simultaneously integrate:
1) Arbitrary asymmetric layer-wise supervision patterns
2) Early exit mechanisms with flexible control
through two independent, composable configuration parameters."

#### 2. ASHEM訓練戦略の新規性

**既存研究との差別化**:
- HAM/HSM: セキュリティ分野のHard example mining（CV/NLP分野とは異なる）
- PLD: Progressive layer addition（Hard example miningとの統合なし）
- **両サーベイ論文**: Early ExitとHard Example Miningの組み合わせに関する記述なし

**ASHEMの独自性**:
- ✅ **Strong Novelty**: Hard Example Mining + Early Exitの統合
- ✅ Two-Phase Training（浅層→深層への段階的展開）
- ✅ Two-Stage Inference（Early Exit）との組み合わせ
- ✅ 言語モデリングへの適用（既存研究は主にCV分野）

**修正されたClaim**:
"ASHEM introduces a novel two-phase training paradigm that:
1) Trains a shallow model on all data
2) Selectively expands to deeper architecture trained exclusively on hard examples identified via confidence thresholds
3) Employs two-stage inference for computational efficiency

This is the first method to combine hard example mining with selective layer expansion and early exit for language modeling."

**注意**: "Progressive Layer Addition"という用語はPLD (NeurIPS 2020)と混同される可能性があるため、"Selective Layer Expansion"を使用することを推奨。

---

### 新規性評価の総括

**✅ 確認された強い新規性**:
1. **Hard Example Mining + Selective Layer Expansion + Early Exit**
   - 両サーベイ論文で組み合わせに関する記述なし
   - 言語モデリングへの適用は本研究が初めて
   - Two-Phase Training（浅層→深層への段階的展開）

2. **2パラメータ統合フレームワーク**
   - 独立かつ組み合わせ可能な2つのパラメータによる統一的制御
   - 既存手法は個別実装が必要
   - 任意の非対称層重みパターンの実現

**⚠️ 既存技術を含む要素**:
1. **Layer-wise Loss Weights**: 既存研究で使用済み（ただし任意パターンは新規）
2. **Early Exit**: 2016年から確立された技術（ただし統合方法は新規）

**📊 実験的検証**:
- WikiText-2 (10K samples)での定量的成果
- Hard examplesへの顕著な改善効果（78% PPL改善）
- 計算効率と精度の両立を実証

---

#### 3. 自動最適化の新規性

**LASHの貢献**:
- `layer_weights`を解析し、最適な実行パスを自動選択
- 最終層のみの場合、8.4%の訓練速度向上
- フレームワークの柔軟性を損なわない最適化

**Claim**: "LASH automatically optimizes execution paths based on layer weight configuration, achieving 8.4% speedup while maintaining full flexibility."

#### 4. 実験結果の新規性

**WikiText-2での検証結果**（10K samples, fc9b140）:
- Hard PPL: **78%改善**（2763 → 668）
- 計算コスト: **36%削減**（64.82% of full model）
- Overall PPL: **15.9%改善**（986 → 830）
- Overall Accuracy: 16.03% → 15.77%（微減）

**既存研究との差別化**:
- **Deep Supervision**: 全層で計算コストが高い（効率性に課題）
- **Early Exit**: 訓練戦略は従来型のまま（Hard examplesへの対応なし）
- **ASHEM**: 訓練と推論の両方を最適化（Hard examplesに特化した段階的訓練）

**重要な知見**:
- Hard examplesへの特化訓練により、難しいサンプルでの性能が大幅向上
- Early Exitによる推論時の計算コスト削減
- 全体の精度を維持しながら効率を改善

### 論文構成案

1. **Introduction**: 統一フレームワークの必要性
   - 既存手法の個別実装の課題
   - 柔軟な戦略カスタマイズの重要性

2. **Related Work**:
   - **Deep Supervision** (Lee et al., 2015)
   - **Discriminative Fine-Tuning** (Howard & Ruder, 2018)
   - **Early Exit Networks** (Teerapittayanon et al., 2016; BranchyNet)
   - **Hard Example Mining** (HAM, HSM等 - 主にCV/セキュリティ分野)
   - **Recent Surveys** (ACM Survey Nov 2024, NLP Survey Jan 2025)
   - **既存手法の課題**: 個別実装、統合の困難さ

3. **LASH Framework**: 2つのコアオプションとアーキテクチャ
   - `layer_weights`: 任意の非対称パターン
   - `routing_threshold`: Early Exit閾値
   - 自動最適化機構

4. **ASHEM Training Strategy**: Hard example miningを活用した新しい訓練戦略
   - Two-Phase Training（浅層→深層）
   - Hard Example Identification（Per-token filtering）
   - Two-Stage Inference

5. **Experiments**: WikiText-2/103でのベースライン比較
   - Standard vs Deep Supervision vs ASHEM
   - Hard examples vs Easy examples の分析
   - 計算コストと精度のトレードオフ

6. **Analysis**:
   - Ablation study（ASHEMの各コンポーネント）
   - Threshold感度分析
   - 計算効率分析（FLOPs, wall-clock time）
   - Scalability検証

7. **Conclusion**: 統一フレームワークの意義と将来展望
   - 新規性の再確認
   - 大規模モデルへの展開可能性

### 今後の実験計画

#### Tier 1（必須実験）

- [ ] WikiText-103での検証（スケーラビリティ）
- [ ] ベースライン比較（Standard, Deep Supervision, Discriminative FT, Early Exit）
- [ ] 中規模モデル（dim=128, layers=4→6）での検証

#### Tier 2（強く推奨）

- [ ] Ablation Study（ASHEMの各コンポーネント）
- [ ] Threshold感度分析（0.7, 0.8, 0.9, 0.95, 0.99）
- [ ] 計算効率分析（FLOPs, wall-clock time）

#### Tier 3（可能であれば）

- [ ] 実際のLLM（Llama等）での検証
- [ ] 他のタスク（分類、要約等）への適用
- [ ] 大規模データセット（C4, The Pile等）での検証

---

## 今後のタスク

- [ ] より大規模なモデルでの検証実験（dim=128, layers=6）
- [ ] 実際の LLM (Llama 等) での検証
- [ ] ASHEM以外の新しい訓練戦略の開発
- [ ] 他のデータセット（C4, The Pile等）での検証
- [ ] Staged DS の実装完成（Per-token filtering の正しい実装）
