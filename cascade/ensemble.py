"""
CASCADEフレームワーク - Ensemble

複数のLLMを統合するクラス。
三角形Attention方式への移行準備中。
"""

from __future__ import annotations

import torch.nn as nn
from typing import List


from .llm import LLM


class Ensemble(nn.Module):
    """
    複数のLLMを統合するアンサンブルモデル。

    アーキテクチャ:
    - 複数のLLM（Hugging Face CausalLM + Early Exit機能）
    - 各LLMは独自のembeddingとlm_headを保持

    設計方針:
        各LLMは独立したモデルとして自己完結する。
        Ensembleはルーティングのみを担当。

    使用例:
        from transformers import AutoModelForCausalLM

        gpt2_0 = AutoModelForCausalLM.from_pretrained('gpt2')
        gpt2_1 = AutoModelForCausalLM.from_pretrained('gpt2')
        llm_0 = LLM(gpt2_0)
        llm_1 = LLM(gpt2_1)
        ensemble = Ensemble([llm_0, llm_1])

    Args:
        llms: LLMインスタンスのリスト
    """

    def __init__(self, llms: List[LLM]):
        super().__init__()

        if len(llms) == 0:
            raise ValueError("少なくとも1つのLLMが必要です")

        # vocab_sizeとdimの一貫性を検証
        vocab_size = llms[0].vocab_size
        dim = llms[0].dim
        for i, llm in enumerate(llms[1:], 1):
            if llm.vocab_size != vocab_size:
                raise ValueError(
                    f"LLM {i}のvocab_size ({llm.vocab_size}) が"
                    f"LLM 0のvocab_size ({vocab_size}) と一致しません"
                )
            if llm.dim != dim:
                raise ValueError(
                    f"LLM {i}のdim ({llm.dim}) が"
                    f"LLM 0のdim ({dim}) と一致しません"
                )

        self.llms = nn.ModuleList(llms)

    @property
    def vocab_size(self) -> int:
        """語彙サイズ（最初のLLMから取得）。"""
        return self.llms[0].vocab_size

    @property
    def dim(self) -> int:
        """モデル次元（最初のLLMから取得）。"""
        return self.llms[0].dim

    @property
    def num_layers(self) -> int:
        """全LLMの合計レイヤー数。"""
        return sum(llm.num_layers for llm in self.llms)

    def add_llm(self, llm: LLM) -> None:
        """
        新しいLLMをアンサンブルに追加。

        Args:
            llm: 追加するLLMインスタンス（vocab_sizeとdimが一致している必要あり）
        """
        if llm.vocab_size != self.vocab_size:
            raise ValueError(
                f"LLMのvocab_size ({llm.vocab_size}) が"
                f"Ensembleのvocab_size ({self.vocab_size}) と一致しません"
            )
        if llm.dim != self.dim:
            raise ValueError(
                f"LLMのdim ({llm.dim}) が"
                f"Ensembleのdim ({self.dim}) と一致しません"
            )
        self.llms.append(llm)
