# HRM Experiment Results

## Overview

This directory contains experimental results for HRM (Hierarchical Reasoning Model) variants on language modeling tasks.

**Note**: すべての学習方法は Universal Training Framework に統一されました。
詳細は [06_universal_framework.md](06_universal_framework.md) を参照。

## Terminology Mapping

本プロジェクトでは既存研究の名称を使用しています：

| This Project (旧) | Academic Term | Reference |
|-------------------|---------------|-----------|
| DeepSupervision | **Deep Supervision** | Lee et al., 2015 |
| DeepSupervision + Early Exit | **DEED** | Tang et al., 2023 |
| Standard Routing | **Auxiliary Loss Training** | Elbayad et al., 2020 |
| Confidence-based Routing | **Early Exit** | Teerapittayanon et al., 2016 |
| Layer-wise Learning Rate | **Discriminative Fine-Tuning** | Howard & Ruder, 2018 |
| Dynamic Alpha | **Learning Rate Curriculum** | Croitoru et al., 2024 |

詳細な引用情報は [REFERENCES.md](../REFERENCES.md) を参照。

## Files

| File | Description |
|------|-------------|
| [01_training_methods.md](01_training_methods.md) | Standard vs DEED comparison |
| [02_layer_analysis.md](02_layer_analysis.md) | Layer-by-layer performance analysis |
| [03_confidence_routing.md](03_confidence_routing.md) | Early Exit (Auxiliary Loss & Deep Supervision) |
| [04_asymmetric_training.md](04_asymmetric_training.md) | Asymmetric Auxiliary Loss with L2 loss comparison |
| [05_summary.md](05_summary.md) | Summary of all results and best practices |
| [06_universal_framework.md](06_universal_framework.md) | **Universal Training Framework** (メイン) |
| [07_limitations.md](07_limitations.md) | Framework の限界と将来拡張 |

## Quick Results

| Rank | Model | PPL | Compute% | vs Standard 3L |
|------|-------|-----|----------|----------------|
| 🥇 | **Discriminative Fine-Tuning (Decreasing LR)** | **18.52** | 65.2% | **46.9% 改善** |
| 🥈 | Discriminative Fine-Tuning (Increasing LR) | 21.14 | 72.1% | 39.3% 改善 |
| 🥉 | Asymmetric Auxiliary Loss (α=0.8) | 22.40 | 65.2% | 35.7% 改善 |
| 4 | Asymmetric Auxiliary Loss (α=0.7) | 22.95 | 65.0% | 34.2% 改善 |
| 5 | Auxiliary Loss Training (α=0.5) | 23.98 | 65.2% | 31.2% 改善 |
| 6 | DEED | 28.13 | 46.6% | 19.3% 改善 |
| 7 | Deep Supervision (3L) | 30.54 | 100% | 12.4% 改善 |
| 8 | Standard (3L) | 34.86 | 100% | (baseline) |

**New in v2.0**: Discriminative Fine-Tuning と Learning Rate Curriculum を追加

## Experimental Setup

- **Dataset**: WikiText-2 (character-level)
- **Train**: 100,000 characters
- **Validation**: 10,000 characters
- **Model**: dim=64, heads=4, layers=3
- **Early Stopping**: Immediate (patience=0)

## Key References

- Lee, C.-Y., et al. (2015). **Deeply-Supervised Nets**. AISTATS 2015. https://arxiv.org/abs/1409.5185
- Tang, Y., et al. (2023). **DEED: Dynamic Early Exit on Decoder**. Amazon Science. https://arxiv.org/abs/2311.08623
- Howard, J., & Ruder, S. (2018). **Universal Language Model Fine-tuning for Text Classification**. ACL 2018. https://arxiv.org/abs/1801.06146
- Elbayad, M., et al. (2020). **Depth-Adaptive Transformer**. ICLR 2020. https://arxiv.org/abs/1910.10073
- Teerapittayanon, S., et al. (2016). **BranchyNet: Fast Inference via Early Exiting**. ICPR 2016. https://arxiv.org/abs/1709.01686
- Croitoru, F.-A., et al. (2024). **Learning Rate Curriculum**. IJCV 2024. https://arxiv.org/abs/2205.09180
