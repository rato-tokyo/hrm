# HRM Project - Claude Instructions

## Project Overview

**EASE: Efficient Asymmetric Supervision for Early-Exit Transformers**

Early-Exit Transformer の学習方法に関する研究プロジェクト。

---

## 論文方針

### フレームワーク名
**EASE** (Efficient Asymmetric Supervision for Early-Exit)

### 論文タイトル案
> "EASE: Efficient Asymmetric Supervision for Early-Exit Transformers"

または

> "Rethinking Auxiliary Loss for Early-Exit Transformers: Why Intermediate Layers Should Not Predict"

### 主要な貢献

1. **中間層損失ゼロの発見**
   - L2（中間層）に損失を適用すると 39.8% 性能悪化
   - 中間層は「純粋な特徴抽出層」として機能させるべき
   - Confidence Calibration の改善につながる

2. **非対称損失重み付け（Asymmetric Auxiliary Loss）**
   - α=0.7（Shallow重視）が最適
   - 既存研究の α=0.5（均等）より 4.3% 改善

3. **Discriminative Fine-Tuning × Early Exit の新組み合わせ**
   - 浅い層に高LR、深い層に低LR
   - 46.9% 改善（最良結果）

4. **Universal Training Framework**
   - Deep Supervision, Auxiliary Loss, Early Exit, Discriminative FT を統一的に表現

### 論文構成

```
1. Introduction
   - Early Exit の重要性と課題
   - 学習方法の体系的研究の不足

2. Universal Training Framework (EASE)
   - 既存手法の統一的表現
   - layer_weights, routing_threshold, layer_lr_scales

3. Experiments
   3.1 中間層損失の影響 (L2=0 vs L2>0)
   3.2 非対称損失重み付け (α探索)
   3.3 Discriminative Fine-Tuning

4. Analysis
   - なぜ中間層損失0が効くか
   - Confidence Calibration との関係

5. Related Work
   - Deep Supervision (Lee et al., 2015)
   - Early Exit (BranchyNet, CALM, LayerSkip)
   - Discriminative Fine-Tuning (ULMFiT)

6. Conclusion
```

### 投稿先候補

| 優先度 | 会議/ジャーナル | 理由 |
|--------|----------------|------|
| 1 | arXiv (プレプリント) | まず公開して反応を見る |
| 2 | EMNLP Findings | 効率的NLPに関心高い |
| 3 | ACL Findings | 同上 |

---

## 用語対応表

| プロジェクト内 (旧) | 学術用語 | Reference |
|-------------------|---------|-----------|
| LPT | **Deep Supervision** | Lee et al., 2015 |
| Standard Routing | **Auxiliary Loss Training** | Elbayad et al., 2020 |
| Confidence-based Routing | **Early Exit** | Teerapittayanon et al., 2016 |
| Layer-wise Learning Rate | **Discriminative Fine-Tuning** | Howard & Ruder, 2018 |
| Dynamic Alpha | **Learning Rate Curriculum** | Croitoru et al., 2024 |

---

## 主要な実験結果

| Rank | Model | PPL | vs Standard 3L |
|------|-------|-----|----------------|
| 🥇 | Discriminative FT (Decreasing LR) | 18.52 | **46.9% 改善** |
| 🥈 | Asymmetric Auxiliary Loss (α=0.7) | 22.95 | 34.2% 改善 |
| 🥉 | Auxiliary Loss (α=0.5) | 23.98 | 31.2% 改善 |
| - | Standard (3L) | 34.86 | (baseline) |

**重要な発見**: L2ロス追加で **39.8% 悪化** (22.95 → 32.07)

---

## 関連研究との位置づけ

| 研究 | 焦点 | EASE との違い |
|------|------|--------------|
| CALM (Google, 2022) | 推論時の判定方法 | 学習時の損失設計に注目 |
| LayerSkip (Meta, 2024) | Layer Dropout + 推論 | 損失の最適配置を発見 |
| EE-LLM (Alibaba, 2023) | スケーラビリティ | 中間層損失0の重要性を発見 |

---

## ファイル構成

```
hrm/
├── CLAUDE.md                    # このファイル
├── src/
│   └── ease/                    # EASE フレームワーク (pip installable)
│       ├── __init__.py          # メインエントリポイント
│       ├── models.py            # StandardTransformer, ConfidenceRoutedTransformer
│       ├── trainer.py           # UniversalConfig, UniversalTrainer, AlphaSchedule
│       └── modules/             # コアモジュール
│           ├── norm.py          # RMSNorm
│           ├── attention.py     # MultiHeadAttention, RoPE
│           ├── ffn.py           # GatedLinearUnit
│           └── transformer.py   # TransformerBlock
├── experiments/
│   ├── __init__.py
│   └── utils.py                 # データ準備、シード設定
├── docs/
│   ├── REFERENCES.md            # 学術的参考文献
│   └── experiments/             # 実験結果ドキュメント
└── run_experiments.py           # 実験実行スクリプト（薄いラッパー）
```

### 使用方法

```python
import sys
sys.path.insert(0, 'src')

from ease import (
    ConfidenceRoutedTransformer,
    UniversalConfig,
    UniversalTrainer,
    PRESETS,
)

# プリセット使用
config = PRESETS['asymmetric']  # α=0.7, L2=0

# カスタム設定
config = UniversalConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},
    routing_threshold=0.95,
)

# モデル・トレーナー作成
model = ConfidenceRoutedTransformer(vocab_size=1000, dim=64, num_layers=3)
trainer = UniversalTrainer(config, vocab_size=1000)
```

---

## 今後のタスク

- [ ] 論文執筆（arXiv 投稿用）
- [ ] より大規模なモデルでの検証実験
- [ ] 実際の LLM (Llama 等) での検証
- [ ] LayerSkip との組み合わせ実験
