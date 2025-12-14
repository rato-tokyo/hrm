# Exit Classifier 方式の比較実験

## 実験日: 2024-12-14

## 概要

exit_classifierの学習ラベル方式と、softmax confidence方式を比較。

## 実験設定

```python
ExperimentConfig(
    dim=64,
    num_heads=4,
    ffn_dim=256,
    max_seq_len=1024,
    causal=True,
    eps=1e-6,
    seq_len=32,
    num_samples=10000,
    block_layers=(2, 2),
)

TrainerConfig(
    batch_size=64,
    max_epochs=50,
    patience=3,
    grad_clip=1.0,
    val_ratio=0.2,
    hard_ratio=0.5,
    lr=1e-3,
    exit_classifier_mode="post",
)
```

## 比較した方式

| 方式 | confidence_mode | exit_label_mode | 説明 |
|------|-----------------|-----------------|------|
| A | exit_classifier | distill | softmax confidenceを蒸留 |
| B | exit_classifier | loss | exp(-loss)を学習 |
| C | softmax | - | max(softmax(logits))を直接使用 |

### ラベル計算の違い

```python
# distill: モデルの確信度（正解不問）
exit_labels = F.softmax(logits, dim=-1).max(dim=-1).values

# loss: 実際の予測精度（正解考慮）
per_token_loss = F.cross_entropy(logits, targets, reduction='none')
exit_labels = torch.exp(-per_token_loss)

# softmax: exit_classifierを使わず、直接計算
confidence = F.softmax(logits, dim=-1).max(dim=-1).values
```

## 結果

### 最終評価

| 指標 | distill | loss | softmax |
|------|---------|------|---------|
| **Final PPL** | 1095.46 | **1071.76** | 1219.52 |
| **Accuracy** | 16.82% | 16.81% | 16.78% |
| **Shallow ratio** | 50.7% | 49.3% | **72.8%** |
| **Compute savings** | 25.3% | 24.6% | **36.4%** |

### 訓練過程

| 指標 | distill | loss | softmax |
|------|---------|------|---------|
| Block 0 val_ppl | 564.95 | 564.95 | 564.95 |
| Block 1 val_ppl | 522.28 | **445.48** | 736.13 |
| Threshold (Block 0) | 0.1738 | 0.0481 | 0.1492 |
| avg_label (Block 0) | 0.184 | 0.074 | - |

### exit_classifier訓練ログ

**distill方式 (Block 0)**:
```
Exit epoch 1/10: loss=0.0385, avg_label=0.184
Exit epoch 10/10: loss=0.0170, avg_label=0.184
```

**loss方式 (Block 0)**:
```
Exit epoch 1/10: loss=0.0509, avg_label=0.074
Exit epoch 10/10: loss=0.0211, avg_label=0.074
```

## 分析

### 1. loss方式がBlock 1訓練で最良

- Block 1 val_ppl: distill(522) < loss(445) < softmax(736)
- loss方式は「正解ラベルを考慮」するため、より正確なhard example収集が可能

### 2. softmax方式は計算効率が最良だが精度が最悪

- Shallow ratio 72.8%（最も多くのトークンがBlock 0でexit）
- しかしBlock 1の訓練が不十分（val_ppl 736）
- 「確信しているが間違っている」トークンを区別できない

### 3. 全方式で最終PPLが訓練時より悪化

- Block 0訓練時: val_ppl ≈ 565
- 最終評価: PPL ≈ 1072-1220

**原因の仮説**:
1. TRUE Early Exit時のexit判定精度
2. hard example repackによる文脈喪失
3. 訓練データと評価データの分布差

## 結論

### 推奨方式

| 目的 | 推奨 |
|------|------|
| **精度重視** | exit_classifier + loss |
| **計算効率重視** | softmax |
| **バランス** | exit_classifier + distill |

### 採用決定

**exit_classifier + loss方式を標準として採用**

理由:
1. Block 1訓練が最も良好（val_ppl 445）
2. 最終PPLが最良（1071）
3. 正解ラベルを考慮した信頼度推定

## 今後の改善案

1. thresholdの調整（exit率を下げる）
2. Block 0でのexit精度の詳細分析
3. hard_ratioの最適化
4. 最終PPL悪化の原因調査
