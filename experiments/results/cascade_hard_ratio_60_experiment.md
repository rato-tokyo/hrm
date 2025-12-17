# CASCADE実験結果: HARD_RATIO=0.6

## 実験概要

**日時**: 2025-12-17 09:43:48
**目的**: HARD_RATIO=0.9での過学習問題を改善するため、0.6に削減して再実験

## 設定

| パラメータ | 値 |
|-----------|-----|
| ベースモデル | HuggingFaceTB/SmolLM2-135M-Instruct |
| 段階あたりのレイヤー数 | 8 |
| Hard token比率 | **60.0%** |
| 段階数 | 5 |
| エポック数 | 10 |
| バッチサイズ | 32 |
| 学習率 | 0.0001 |
| Early stopping patience | 1 |
| 訓練サンプル数 | 1000 |
| 検証サンプル数 | 100 |
| デバイス | cuda (Colab T4 GPU) |

## ベースモデル情報

- パラメータ数: 134,515,008 (134.5M)
- レイヤー数: 30
- 訓練トークン数: 128,000
- 検証トークン数: 12,800

## 結果サマリー

| 段階 | 追加層 | 合計層 | 訓練tokens | Val PPL | 累計パラメータ |
|------|--------|--------|------------|---------|---------------|
| 1 | 8 | 38 | 76,800 | 155.41 | 191.1M |
| 2 | 8 | 46 | 46,080 | 115.91 | 247.8M |
| 3 | 8 | 54 | 27,648 | 48.14 | 304.4M |
| 4 | 8 | 62 | 16,589 | 19.54 | 361.0M |
| 5 | 8 | 70 | 9,953 | 11.91 | 417.7M |

## 各段階の詳細

### Stage 1

| 項目 | 値 |
|------|-----|
| 閾値 (cos_sim) | 0.1234 |
| 訓練Hard tokens | 76,800 / 128,000 (60.0%) |
| 検証Hard tokens | 7,680 |
| 追加パラメータ | 56.6M |

**訓練ログ**:
```
Epoch 1/10: train_loss=5.7942, val_loss=5.3436, val_ppl=209.27, time=130.0s
Epoch 2/10: train_loss=3.6297, val_loss=5.0232, val_ppl=151.90, time=129.6s
Epoch 3/10: train_loss=2.3241, val_loss=5.0461, val_ppl=155.41, time=129.8s
Early stopping at epoch 3 (patience=1)
```

**最良val_ppl**: 151.90 (Epoch 2)
**訓練時間**: 389.5秒

### Stage 2

| 項目 | 値 |
|------|-----|
| 閾値 (cos_sim) | 0.3492 |
| 訓練Hard tokens | 46,080 / 76,800 (60.0%) |
| 検証Hard tokens | 4,608 |
| 追加パラメータ | 56.6M |

**訓練ログ**:
```
Epoch 1/10: train_loss=4.1996, val_loss=4.6793, val_ppl=107.70, time=78.0s
Epoch 2/10: train_loss=2.1855, val_loss=4.6294, val_ppl=102.45, time=77.9s
Epoch 3/10: train_loss=1.3176, val_loss=4.7528, val_ppl=115.91, time=77.8s
Early stopping at epoch 3 (patience=1)
```

**最良val_ppl**: 102.45 (Epoch 2)
**訓練時間**: 233.8秒

### Stage 3

| 項目 | 値 |
|------|-----|
| 閾値 (cos_sim) | 0.3730 |
| 訓練Hard tokens | 27,648 / 46,080 (60.0%) |
| 検証Hard tokens | 2,765 |
| 追加パラメータ | 56.6M |

**訓練ログ**:
```
Epoch 1/10: train_loss=2.3030, val_loss=3.8226, val_ppl=45.72, time=46.9s
Epoch 2/10: train_loss=1.0141, val_loss=3.8742, val_ppl=48.14, time=46.9s
Early stopping at epoch 2 (patience=1)
```

**最良val_ppl**: 45.72 (Epoch 1)
**訓練時間**: 93.8秒

### Stage 4

| 項目 | 値 |
|------|-----|
| 閾値 (cos_sim) | 0.2975 |
| 訓練Hard tokens | 16,589 / 27,648 (60.0%) |
| 検証Hard tokens | 1,659 |
| 追加パラメータ | 56.6M |

**訓練ログ**:
```
Epoch 1/10: train_loss=1.1144, val_loss=2.9473, val_ppl=19.05, time=28.2s
Epoch 2/10: train_loss=0.5184, val_loss=2.9473, val_ppl=19.05, time=28.3s
Epoch 3/10: train_loss=0.4424, val_loss=2.9722, val_ppl=19.54, time=28.1s
Early stopping at epoch 3 (patience=1)
```

**最良val_ppl**: 19.05 (Epoch 1-2)
**訓練時間**: 84.6秒

### Stage 5

| 項目 | 値 |
|------|-----|
| 閾値 (cos_sim) | 0.1982 |
| 訓練Hard tokens | 9,953 / 16,589 (60.0%) |
| 検証Hard tokens | 995 |
| 追加パラメータ | 56.6M |

**訓練ログ**:
```
Epoch 1/10: train_loss=1.0462, val_loss=2.4627, val_ppl=11.74, time=17.0s
Epoch 2/10: train_loss=0.4875, val_loss=2.4454, val_ppl=11.54, time=17.0s
Epoch 3/10: train_loss=0.4149, val_loss=2.4774, val_ppl=11.91, time=16.9s
Early stopping at epoch 3 (patience=1)
```

**最良val_ppl**: 11.54 (Epoch 2)
**訓練時間**: 51.0秒

## 分析

### HARD_RATIO=0.9との比較

| 指標 | HARD_RATIO=0.9 | HARD_RATIO=0.6 |
|------|---------------|---------------|
| Stage 1 訓練tokens | 115,200 | 76,800 |
| Stage 1 最良val_ppl | 84.43 | 151.90 |
| Stage 2 最良val_ppl | 92.65 (悪化) | 102.45 (改善) |
| 全5段階完了 | ❌ (Stage 3で中断) | ✅ |

### 重要な観察

1. **過学習は依然として存在するが、程度は軽減**
   - 各Stageでtrain_lossは急減、val_lossは微減または横ばい
   - ただし0.9ほど顕著ではない

2. **各Stageのval_pplは単体では意味が限定的**
   - Stage 1: 151.90 → Stage 5: 11.54
   - **これはhard tokensのみでの評価**であり、全トークンでのEnsemble評価ではない

3. **訓練トークン数の減少**
   ```
   Stage 1: 76,800 (60%)
   Stage 2: 46,080 (60% of 76,800)
   Stage 3: 27,648
   Stage 4: 16,589
   Stage 5: 9,953
   ```
   各段階で60%に減少 → 最終段階は最初の約7.8%のみ

4. **閾値の推移**
   ```
   Stage 1: 0.1234 (低い)
   Stage 2: 0.3492 (上昇)
   Stage 3: 0.3730 (微上昇)
   Stage 4: 0.2975 (下降)
   Stage 5: 0.1982 (下降)
   ```
   Stage 3-4で閾値が下がる = hidden statesの変化が大きくなっている（興味深い）

### 訓練時間

| 段階 | 訓練時間 |
|------|----------|
| Stage 1 | 389.5秒 |
| Stage 2 | 233.8秒 |
| Stage 3 | 93.8秒 |
| Stage 4 | 84.6秒 |
| Stage 5 | 51.0秒 |
| **合計** | **852.7秒 (約14分)** |

## 次のステップ

1. **Ensemble評価スクリプトの実行**
   ```bash
   python experiments/evaluate_cascade_ensemble.py outputs/cascade_20251217_094348
   ```
   各パターン（Base only, Base+Stage1, ...）での全トークンval_pplを測定

2. **ベースライン実験との比較**
   ```bash
   python experiments/baseline_360m_finetuning.py
   ```
   SmolLM2-360M単体のファインチューニング結果と比較

## パラメータ効率性

| 構成 | パラメータ数 | 備考 |
|------|-------------|------|
| Base (SmolLM2-135M) | 134.5M | フリーズ |
| Base + Stage1 | 191.1M | +56.6M |
| Base + Stage1-5 | 417.7M | +283.2M |
| SmolLM2-360M (ベースライン) | 360M | 全パラメータ訓練可能 |

CASCADEの最終構成(417.7M)はSmolLM2-360M(360M)より大きいが、
ベースの134.5Mはフリーズされており、訓練パラメータは283.2Mのみ。
