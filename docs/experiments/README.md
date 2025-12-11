# HRM Experiment Results

## Overview

This directory contains experimental results for HRM (Hierarchical Reasoning Model) variants on language modeling tasks.

**Note**: すべての学習方法は Universal Training Framework に統一されました。
詳細は [06_universal_framework.md](06_universal_framework.md) を参照。

## Files

| File | Description |
|------|-------------|
| [01_training_methods.md](01_training_methods.md) | Standard vs LPT training comparison (アルゴリズム解説) |
| [02_layer_analysis.md](02_layer_analysis.md) | Layer-by-layer performance analysis |
| [03_confidence_routing.md](03_confidence_routing.md) | Confidence-Routed Transformer (Standard & LPT) |
| [04_asymmetric_training.md](04_asymmetric_training.md) | Asymmetric training with L2 loss comparison |
| [05_summary.md](05_summary.md) | Summary of all results and best practices |
| [06_universal_framework.md](06_universal_framework.md) | **Universal Training Framework** (メイン) |
| [07_limitations.md](07_limitations.md) | Framework の限界と将来拡張 |

## Quick Results

| Rank | Model | PPL | Compute% | vs Standard 3L |
|------|-------|-----|----------|----------------|
| 🥇 | **Asymmetric (α=0.7)** | **22.95** | 65.0% | **34.2% 改善** |
| 🥈 | Standard Routing (α=0.5) | 23.98 | 65.2% | 31.2% 改善 |
| 🥉 | LPT Routing | 28.13 | 46.6% | 19.3% 改善 |
| 4 | LPT (3L) | 30.54 | 100% | 12.4% 改善 |
| 5 | Standard (3L) | 34.86 | 100% | (baseline) |

## Experimental Setup

- **Dataset**: WikiText-2 (character-level)
- **Train**: 100,000 characters
- **Validation**: 10,000 characters
- **Model**: dim=64, heads=4, layers=3
- **Early Stopping**: Immediate (patience=0)
