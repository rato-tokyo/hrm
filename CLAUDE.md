# HRM Project - Claude Instructions

## Project Overview

**LASH: Layered Adaptive Supervision Hierarchy**

層を組み合わせる柔軟なフレームワーク。3つのコアオプションで全てを制御。

---

## フレームワーク概要

### コアコンセプト: 層の柔軟な組み合わせ

**LASH**の3つのコアオプションで全てを制御：

| オプション | 説明 | Reference |
|-----------|------|-----------|
| **layer_weights** | 層ごとの損失重み（どの層で学習するか） | - |
| **layer_lr_scales** | 層ごとの学習率スケール | Howard & Ruder, 2018 |
| **routing_threshold** | 推論時Early Exit閾値 | Teerapittayanon et al., 2016 |

**重要**: StandardとDeep Supervisionは単なる設定パターンの違い。同じフレームワークで実現可能。

### 設定例：柔軟な組み合わせ

#### パターン1: Standard Transformer（従来型LLM）
```python
config = TrainingConfig(
    layer_weights={1: 0, 2: 0, 3: 1}  # 最終層のみ
)
```

#### パターン2: Deep Supervision（全層均等）
```python
config = TrainingConfig(
    layer_weights={1: 0.33, 2: 0.33, 3: 0.33}  # 全層で学習
)
```

#### パターン3: Asymmetric + Early Exit（柔軟な設定）
```python
# 6層モデル、Layer 3と6でのみ損失計算 + Early Exit
config = TrainingConfig(
    layer_weights={
        1: 0,    # 学習しない
        2: 0,    # 学習しない
        3: 0.5,  # 学習する（Early Exit候補層）
        4: 0,    # 学習しない
        5: 0,    # 学習しない
        6: 0.5   # 学習する（最終層）
    },
    routing_threshold=0.95,  # 推論時、Layer 3で95%信頼度あれば終了
    exit_layer=3
)
```

#### パターン4: Discriminative Fine-Tuning（層ごとの学習率）
```python
config = TrainingConfig(
    layer_weights={1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1},  # 最終層のみ
    layer_lr_scales={1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2, 6: 0.1}  # 深い層ほど低学習率
)
```

#### パターン5: 完全カスタム（全オプション活用）
```python
config = TrainingConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3, 4: 0, 5: 0, 6: 0},  # Layer 1重視
    layer_lr_scales={1: 1.0, 2: 0.5, 3: 0.1, 4: 0.05, 5: 0.02, 6: 0.01},
    routing_threshold=0.9,  # Layer 1で90%信頼度あれば終了
    exit_layer=1
)
```

---

## ファイル構成

```
hrm/
├── CLAUDE.md                    # このファイル
├── src/
│   └── ease/                    # LASH フレームワーク（ディレクトリ名は互換性のためease）
│       ├── __init__.py          # メインエントリポイント
│       ├── models.py            # Standard, DeepSupervision Transformer
│       ├── trainer.py           # TrainingConfig, Trainer, ASHEMConfig
│       └── modules/             # コアモジュール
│           ├── norm.py          # RMSNorm
│           ├── attention.py     # MultiHeadAttention, RoPE
│           ├── ffn.py           # GatedLinearUnit
│           └── transformer.py   # TransformerBlock
├── experiments/
│   ├── __init__.py
│   └── utils.py                 # データローダー、実験ユーティリティ
├── docs/
│   ├── PAPER_DIRECTION.md       # 論文の方向性
│   └── experiments/             # 実験結果ドキュメント
│       ├── hard_example_mining.md
│       └── progressive_layer_training.md
└── colab2.py                    # ASHEM実験（Colab実行用メインスクリプト）
```

---

## 使用方法

### 基本的な使用例

```python
import sys
sys.path.insert(0, 'src')

from ease import DeepSupervisionTransformer, Trainer, TrainingConfig

# モデル作成
model = DeepSupervisionTransformer(vocab_size=1000, dim=64, num_layers=3)

# 設定: LASHの3つのオプションで全てを制御
config = TrainingConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},        # 層ごとの損失重み
    layer_lr_scales={1: 1.0, 2: 0.5, 3: 0.1},    # 層ごとの学習率
    routing_threshold=0.95,                       # Early Exit閾値
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
# → layer_weights={1: 0, 2: 0, 3: 1}

# Deep Supervision設定（全層均等）
config = create_deep_supervision_config(num_layers=3)
# → layer_weights={1: 0.33, 2: 0.33, 3: 0.33}
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

## 訓練戦略

LASHフレームワークは4つの訓練戦略をサポート：

### 1. Standard
最終層のみで学習（従来のLLM訓練）
```python
config = TrainingConfig(layer_weights={1: 0, 2: 0, 3: 1})
```

### 2. Deep Supervision
全層で均等に学習
```python
config = TrainingConfig(layer_weights={1: 0.33, 2: 0.33, 3: 0.33})
```

### 3. Discriminative Fine-Tuning
層ごとの学習率設定（Howard & Ruder, 2018）
```python
config = TrainingConfig(
    layer_weights={1: 0, 2: 0, 3: 1},
    layer_lr_scales={1: 1.0, 2: 0.8, 3: 0.6}
)
```

### 4. ASHEM (Adaptive Supervision via Hard Example Mining)
Hard examplesに特化した段階的訓練
- **Phase 1**: 浅層モデル（2層）で全データ訓練
- **Phase 2**: 深層モデル（4層）でHard examplesのみ訓練
- **推論**: Two-stage routing（Early Exit）
- **実験結果**: Hard PPL 78%改善、計算コスト36%削減

```python
from ease import ASHEMConfig

ashem_config = ASHEMConfig(
    phase1_layers=2,
    hard_example_ratio=0.5,
    phase2_layers=4,
)
```

詳細: [docs/experiments/hard_example_mining.md](docs/experiments/hard_example_mining.md)

---

## パフォーマンス最適化

### compute_loss() の自動最適化

**LASHの3つのオプションを完全に維持したまま、訓練速度を最適化**:

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
- ✅ `layer_lr_scales`: 独立（optimizer側で処理）
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
- Teerapittayanon et al. (2016) - Early Exit

### ASHEM Training Strategy
- **ASHEM**: Adaptive Supervision via Hard Example Mining（本研究）
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
- Progressive Layer Addition: Related to PLD (NeurIPS 2020)

---

---

## 実験実行原則

### Google Colab実行を前提

**重要**: しばらくの間、すべての実験はGoogle Colabで実行します。

#### 理由
- GPU（NVIDIA L4等）の利用可能性
- 大規模データセット（WikiText-2等）の高速処理
- 長時間訓練の安定実行

#### 実行スクリプト
- **メイン実験**: `colab2.py` (ASHEM実験)
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

---

## 今後のタスク

- [ ] より大規模なモデルでの検証実験（dim=128, layers=6）
- [ ] 実際の LLM (Llama 等) での検証
- [ ] ASHEM以外の新しい訓練戦略の開発
- [ ] 他のデータセット（C4, The Pile等）での検証
