"""
CASCADEフレームワーク - Ensemble

複数のLLMを統合し、TRUE Early Exitによるルーティングを管理。

注意: evaluate()は評価専用。訓練はCascadeTrainerを使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple


from .llm import LLM
from .cascade_dataset import (
    create_cascade_dataset,
    get_dataset_info,
)


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

    def evaluate(
        self,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        【評価用】TRUE Early Exitで評価。

        各LLMが自身のexit/continueを判定し、統計を計算。
        継続トークンは次LLMに渡される。

        Args:
            val_batches: 検証用の(x, y)バッチのリスト
            batch_size: 評価時のバッチサイズ

        Returns:
            ppl, accuracy, llm_stats, total_tokensなどを含むDict
        """
        device = next(self.parameters()).device
        self.eval()

        # 最初のLLMでtoken_idsからhidden statesに変換
        first_llm = self.llms[0]
        all_hidden: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                h_out, _ = first_llm.forward(x, input_type="token_ids")
                all_hidden.append(h_out.cpu())
                all_targets.append(y.cpu())

        # 最初のLLMの評価用Dataset作成
        hidden_cat = torch.cat(all_hidden)
        targets_cat = torch.cat(all_targets)
        current_dataset = create_cascade_dataset(hidden_cat, targets_cat)
        info = get_dataset_info(current_dataset)
        total_tokens = info['num_tokens']

        from .llm_evaluator import evaluate_llm

        llm_stats: List[Dict[str, Any]] = []
        total_loss = 0.0
        total_correct = 0

        # 最初のLLMの統計を計算
        is_last = (len(self.llms) == 1)
        continue_dataset, stats = evaluate_llm(first_llm, current_dataset, batch_size, is_last=is_last)
        llm_stats.append(stats)
        total_loss += stats['loss']
        total_correct += stats['correct']

        current_dataset = continue_dataset

        # 2番目以降のLLMで評価
        for llm_idx, llm in enumerate(self.llms[1:], 1):
            info = get_dataset_info(current_dataset)
            if info['num_sequences'] == 0:
                llm_stats.append({
                    'loss': 0.0,
                    'correct': 0,
                    'input_tokens': 0,
                    'exit_tokens': 0,
                    'layers_computed': 0,
                })
                continue

            is_last = (llm_idx == len(self.llms) - 1)
            continue_dataset, stats = evaluate_llm(llm, current_dataset, batch_size, is_last=is_last)

            llm_stats.append(stats)
            total_loss += stats['loss']
            total_correct += stats['correct']

            current_dataset = continue_dataset

        total_layers_computed = sum(s['layers_computed'] for s in llm_stats)
        max_layers_computed = total_tokens * self.num_layers

        # PPLとAccuracyを計算
        ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        return {
            'ppl': ppl,
            'accuracy': accuracy,
            'llm_stats': llm_stats,
            'total_tokens': total_tokens,
            'total_layers_computed': total_layers_computed,
            'max_layers_computed': max_layers_computed,
        }
