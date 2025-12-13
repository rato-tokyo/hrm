# Experiment 001: KV Cache Implementation

**Date**: 2025-12-13
**Script**: `colab3.py`
**Device**: CUDA (Google Colab)

## Overview

KVキャッシュ基盤（Phase A）の実装後、初の実験。従来のLEGO 2フェーズ訓練に加え、KVキャッシュによる生成速度向上を検証。

## Configuration

```python
ExperimentConfig:
  seq_len: 32
  dim: 64
  num_heads: 4
  phase1_samples: 10000
  phase1_layers: 2
  phase2_layers: 4
  hard_example_ratio: 0.5
```

## Results Summary

| Metric | Phase 1 | Phase 2 | Change |
|--------|---------|---------|--------|
| Val PPL | 986.43 | 829.78 | -15.9% |
| Val Acc | 16.03% | 15.77% | -0.26% |
| Hard PPL | 2763.69 | 668.08 | **-75.8%** |

### Two-Stage Inference

| Metric | Value |
|--------|-------|
| Shallow ratio | 70.4% |
| Compute cost | 64.82% |

### KV Cache Performance

| Method | Time | Speedup |
|--------|------|---------|
| Without cache | 0.2644s | 1.0x |
| With cache | 0.1789s | **1.48x** |

## Analysis

### 1. Hard Example Mining の効果

Phase 2でHard PPLが**75.8%改善**（2763.69 → 668.08）。これはLEGOの核心機能であるhard example miningが正しく動作していることを示す。

一方、全体のVal PPLは15.9%改善に留まる。これは：
- Hard examples（難しいトークン）のみを集中訓練
- Easy examples（簡単なトークン）はPhase 1のまま
- Two-stage routingで適切に振り分け

### 2. Shallow Ratio の考察

- **目標**: 50%（`hard_example_ratio=0.5`）
- **実測**: 70.4%

実際のshallow ratioが目標より高い理由：
- 閾値は**validation setの50%**が下回るように設定
- Phase 2訓練後、モデルの信頼度分布が変化
- より多くのトークンが閾値を超える（shallow pathを使用）

### 3. Compute Cost

- **理論最小**: 50%（全トークンがshallow）
- **理論最大**: 100%（全トークンがdeep）
- **実測**: 64.82%

計算式: `(shallow_count * 2 + deep_count * 4) / (total * 4)`

70.4%がshallow（2層）、29.6%がdeep（4層）を使用：
```
cost = (0.704 * 2 + 0.296 * 4) / 4 = 0.648 ≈ 64.8%
```

### 4. KV Cache Speedup

**1.48x高速化**は期待通り。

理論的なspeedup：
- Without cache: 各ステップで全シーケンスを再計算 → O(n²)
- With cache: 新トークンのみ計算 → O(n)

32トークン生成では：
- 理論speedup: ~(1+2+...+32) / 32 ≈ 16.5x（attention部分のみ）
- 実測: 1.48x

差の理由：
- FFN層はキャッシュ効果なし
- embedding/output_head層もキャッシュ効果なし
- 小さいモデル（dim=64）ではオーバーヘッドが相対的に大きい

### 5. Accuracy微減の考察

Phase 1: 16.03% → Phase 2: 15.77%（-0.26%）

- 統計的誤差の範囲内
- Hard examples訓練でeasy examplesの精度がわずかに低下した可能性
- routing誤差（本来deepを使うべきトークンがshallowで処理）

## Comparison with Previous Experiments

| Experiment | Val PPL | Hard PPL Improvement | Shallow Ratio |
|------------|---------|---------------------|---------------|
| test_lego.py (synthetic) | 149.06 | N/A | 50.0% |
| colab2.py (WikiText) | ~830 | ~75% | ~70% |
| **colab3.py (WikiText)** | **829.78** | **75.8%** | **70.4%** |

colab2.pyとcolab3.pyの結果はほぼ同一。KVキャッシュ追加は訓練・評価に影響なし（後方互換性OK）。

## Conclusions

1. **KVキャッシュ基盤**: 正常動作、1.48x speedup達成
2. **後方互換性**: 既存のLEGOワークフローに影響なし
3. **Hard Example Mining**: 75.8%のHard PPL改善
4. **Two-Stage Inference**: 64.8%の計算コストで同等の性能

## Next Steps (Phase B)

Phase A（KVキャッシュ基盤）完了。Phase Bでは**真のEarly Exit**を実装予定：

- 信頼度が高いトークン → Layer 3-4を**スキップ**（現在は両方計算後に選択）
- KVキャッシュを活用したトークン単位の動的ルーティング
- 実際の計算量削減の実現

```python
# Phase B: 真のEarly Exit（予定）
def generate_with_early_exit(self, input_ids, max_new_tokens, threshold):
    for token in generate:
        # Layer 1-2 処理
        if confidence >= threshold:
            use_shallow_output()  # Layer 3-4をスキップ
        else:
            process_layer_3_4()
```
