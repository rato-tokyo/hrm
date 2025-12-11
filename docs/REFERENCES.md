# References / 参考文献

本プロジェクトで使用している技術の学術的参考文献です。

---

## Core Techniques / コア技術

### 1. Deep Supervision (深層監督)

中間層に補助分類器を付与して学習を行う手法。

**Primary Reference:**
- Lee, C.-Y., Xie, S., Gallagher, P., Zhang, Z., & Tu, Z. (2015). **Deeply-Supervised Nets**. In *Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
  - URL: https://arxiv.org/abs/1409.5185

**Follow-up Works:**
- Huang, G., et al. (2017). **Multi-Scale Dense Networks for Resource Efficient Image Classification (MSDNet)**. In *ICLR 2018*.
  - URL: https://arxiv.org/abs/1703.09844

---

### 2. Early Exit / Adaptive Computation

推論時に中間層から早期に出力を行う手法。

**Survey:**
- Matsubara, Y., et al. (2024). **Early-Exit Deep Neural Network - A Comprehensive Survey**. In *ACM Computing Surveys*.
  - URL: https://dl.acm.org/doi/10.1145/3698767

**Key Papers:**
- Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). **BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks**. In *ICPR 2016*.
  - URL: https://arxiv.org/abs/1709.01686

- Schwartz, R., et al. (2020). **The Right Tool for the Job: Matching Model and Instance Complexities**. In *ACL 2020*.
  - URL: https://arxiv.org/abs/2004.07453

- Schuster, T., et al. (2022). **Confident Adaptive Language Modeling (CALM)**. In *NeurIPS 2022*.
  - URL: https://arxiv.org/abs/2207.07061

---

### 3. Discriminative Fine-Tuning / Layer-wise Learning Rate

層ごとに異なる学習率を適用する手法。

**Primary Reference (ULMFiT):**
- Howard, J., & Ruder, S. (2018). **Universal Language Model Fine-tuning for Text Classification**. In *ACL 2018*.
  - URL: https://arxiv.org/abs/1801.06146
  - Key technique: Discriminative Fine-Tuning, Slanted Triangular Learning Rates, Gradual Unfreezing

**Layer-wise Learning Rate Decay (LLRD):**
- Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). **How to Fine-Tune BERT for Text Classification**. In *CCL 2019*.
  - URL: https://arxiv.org/abs/1905.05583

- Mosbach, M., Andriushchenko, M., & Klakow, D. (2021). **On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines**. In *ICLR 2021*.
  - URL: https://arxiv.org/abs/2006.04884

---

### 4. LeRaC: Learning Rate Curriculum

入力に近い層ほど高い学習率を設定し、徐々に統一していくカリキュラム学習手法。

**Primary Reference:**
- Croitoru, F.-A., Ristea, N.-C., Ionescu, R. T., & Sebe, N. (2024). **Learning Rate Curriculum**. In *International Journal of Computer Vision*.
  - URL: https://arxiv.org/abs/2205.09180
  - GitHub: https://github.com/CroitoruAlin/LeRaC

---

### 5. Progressive Layer Training

層を段階的に追加・学習する手法。

**Progressive Layer Dropping:**
- Zhang, M., et al. (2020). **Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping**. In *NeurIPS 2020*.
  - URL: https://arxiv.org/abs/2010.13369
  - DeepSpeed Implementation: https://www.deepspeed.ai/tutorials/progressive_layer_dropping/

**Progressive Growing:**
- Gong, L., et al. (2019). **Efficient Training of BERT by Progressively Stacking**. In *ICML 2019*.
  - URL: https://arxiv.org/abs/1910.02697

---

### 6. Auxiliary Loss Training

Early Exitのための補助損失を使用した学習手法。

**Key Papers:**
- Elbayad, M., Gu, J., Grave, E., & Auli, M. (2020). **Depth-Adaptive Transformer**. In *ICLR 2020*.
  - URL: https://arxiv.org/abs/1910.10073

- Xin, J., et al. (2020). **DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference**. In *ACL 2020*.
  - URL: https://arxiv.org/abs/2004.12993

---

## Terminology Mapping / 用語対応表

| This Project (旧) | Standard Term (既存名称) | Reference |
|-------------------|-------------------------|-----------|
| LPT (Layer-wise Progressive Training) | **Deep Supervision** | Lee et al., 2015 |
| Layer-wise Learning Rate | **Discriminative Fine-Tuning** / **LLRD** | Howard & Ruder, 2018 |
| Confidence-based Routing | **Early Exit** | Teerapittayanon et al., 2016 |
| Dynamic α | **Learning Rate Curriculum** | Croitoru et al., 2024 |
| Asymmetric Training | **Auxiliary Loss Training** | Elbayad et al., 2020 |
| Routing Threshold | **Confidence Threshold** | Schwartz et al., 2020 |

---

## Our Contributions / 本プロジェクトの貢献

既存技術を組み合わせつつ、以下の新しい知見を発見：

1. **Intermediate Layer Loss Elimination**: 中間層（L2）の損失を0にすることで39.8%の性能改善
2. **Asymmetric Loss Weighting**: α=0.7（浅い層重視）が最適
3. **Discriminative Fine-Tuning for Early Exit**: Early Exitモデルへの層別学習率適用で46.9%改善
4. **Unified Framework**: Deep Supervision、Early Exit、Discriminative Fine-Tuningを統一的に表現

---

## Citation Format

```bibtex
@article{lee2015deeply,
  title={Deeply-Supervised Nets},
  author={Lee, Chen-Yu and Xie, Saining and Gallagher, Patrick and Zhang, Zhengyou and Tu, Zhuowen},
  journal={Proceedings of AISTATS},
  year={2015}
}

@inproceedings{howard2018universal,
  title={Universal Language Model Fine-tuning for Text Classification},
  author={Howard, Jeremy and Ruder, Sebastian},
  booktitle={Proceedings of ACL},
  year={2018}
}

@article{croitoru2024lerac,
  title={Learning Rate Curriculum},
  author={Croitoru, Florinel-Alin and Ristea, Nicolae-Catalin and Ionescu, Radu Tudor and Sebe, Nicu},
  journal={International Journal of Computer Vision},
  year={2024}
}

@inproceedings{schuster2022calm,
  title={Confident Adaptive Language Modeling},
  author={Schuster, Tal and Fisch, Adam and Gupta, Jai and Dehghani, Mostafa and Bahri, Dara and Tran, Vinh Q and Tay, Yi and Metzler, Donald},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
