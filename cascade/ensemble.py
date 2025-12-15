"""
CASCADEフレームワーク - Ensemble（評価専用）

複数のLLMを統合し、TRUE Early Exitによるルーティングを管理。

注意: このクラスのevaluate()は評価専用。訓練はensemble_trainer.pyを使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, List, Any

from .llm import LLM


class Ensemble(nn.Module):
    """
    複数のLLMを統合するアンサンブルモデル。

    アーキテクチャ:
    - Embeddingレイヤー（トークン → hidden states）
    - 複数のLLM（Early Exit機能付き）
    - 共有output head（hidden states → logits）

    管理項目:
    - TRUE Early ExitによるLLM間ルーティング
    - exit統計の計算

    使用例:
        # LLMをラップして統合
        llm_0 = LLM(pretrained_transformer)
        llm_1 = LLM(TransformerBlock(...))

        ensemble = Ensemble(vocab_size, dim, [llm_0, llm_1])

        # 訓練後、LLMを追加
        llm_2 = LLM(TransformerBlock(...))
        ensemble.add_llm(llm_2)

    Args:
        vocab_size: 語彙サイズ
        dim: モデル次元（embedding次元）
        llms: LLMインスタンスのリスト
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        llms: List[LLM],
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.llms = nn.ModuleList(llms)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # 全LLMでOutput headを共有
        for llm in self.llms:
            llm.set_output_head(self.output_head)

        self._init_weights()

    def _init_weights(self) -> None:
        """EmbeddingとOutput headの重みを初期化。"""
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.output_head.weight, std=1.0 / math.sqrt(self.dim))

    @property
    def num_layers(self) -> int:
        """全LLMの合計レイヤー数。"""
        return sum(llm.num_layers for llm in self.llms)

    def add_llm(self, llm: LLM) -> None:
        """
        新しいLLMをアンサンブルに追加。

        追加されたLLMには共有output_headが自動的に設定される。

        Args:
            llm: 追加するLLMインスタンス
        """
        llm.set_output_head(self.output_head)
        self.llms.append(llm)

    def evaluate(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        【評価用】TRUE Early Exitによる推論。

        各トークンはcos_sim >= thresholdとなった最初のLLMでexitする。
        exitしたトークンは後続LLMを**実際に通過しない**（TRUE early exit）。

        継続トークンのみを新シーケンスに再構成して次LLMに渡す。
        これにより計算量が実際に削減される。

        evaluator.pyのevaluate_ensemble()から呼ばれる。

        Args:
            x: 入力トークンID (batch_size, seq_len)

        Returns:
            (logits, stats)のタプル
            - logits: (batch_size, seq_len, vocab_size)
            - stats: exit_counts, shallow_ratio, compute_costを含むDict
        """
        batch_size, seq_len = x.shape
        device = x.device
        total_tokens = batch_size * seq_len

        # Embedding
        h = self.embedding(x)  # (batch_size, seq_len, dim)

        # 全トークンをフラットに管理
        h_flat = h.view(total_tokens, self.dim)  # (total_tokens, dim)

        # 各トークンの元の位置を記録（最終的な並べ替え用）
        original_indices = torch.arange(total_tokens, device=device)

        # 結果格納用
        final_logits_flat = torch.zeros(total_tokens, self.vocab_size, device=device)
        exit_llm_indices = torch.full((total_tokens,), -1, dtype=torch.long, device=device)

        # 現在処理中のトークンのインデックス
        active_indices = original_indices.clone()
        active_h = h_flat.clone()

        for llm_idx, llm in enumerate(self.llms):
            if active_h.shape[0] == 0:
                break

            is_last_llm = (llm_idx == len(self.llms) - 1)
            num_active = active_h.shape[0]

            # シーケンスに再構成してLLM処理（Attentionのため）
            # 継続トークンを新しいシーケンスとして詰め直す
            num_sequences = (num_active + seq_len - 1) // seq_len
            padded_len = num_sequences * seq_len

            if num_active < padded_len:
                # パディングが必要
                padding = torch.zeros(padded_len - num_active, self.dim, device=device)
                h_padded = torch.cat([active_h, padding], dim=0)
            else:
                h_padded = active_h

            h_seq = h_padded.view(num_sequences, seq_len, self.dim)

            # LLM処理（評価モード）
            h_out_seq, logits_seq, should_exit_seq = llm.evaluate(h_seq)

            # フラットに戻す（パディング部分を除去）
            h_out_flat = h_out_seq.view(-1, self.dim)[:num_active]
            logits_flat = logits_seq.view(-1, self.vocab_size)[:num_active]
            should_exit_flat = should_exit_seq.view(-1)[:num_active]

            if is_last_llm:
                # 最終LLM: 全ての残りトークンをここでexit
                final_logits_flat[active_indices] = logits_flat
                exit_llm_indices[active_indices] = llm_idx
            else:
                # exitするトークンのlogitsを保存
                exit_mask = should_exit_flat
                exit_indices = active_indices[exit_mask]
                final_logits_flat[exit_indices] = logits_flat[exit_mask]
                exit_llm_indices[exit_indices] = llm_idx

                # 継続トークンのみを次LLMへ
                continue_mask = ~exit_mask
                active_indices = active_indices[continue_mask]
                active_h = h_out_flat[continue_mask]

        # 元の形状に戻す
        final_logits = final_logits_flat.view(batch_size, seq_len, self.vocab_size)

        exit_counts = [
            int((exit_llm_indices == i).sum().item()) for i in range(len(self.llms))
        ]
        return final_logits, self._compute_exit_stats(exit_counts)

    def _compute_exit_stats(self, exit_counts: List[int]) -> Dict[str, Any]:
        """exit統計を計算。

        Args:
            exit_counts: 各LLMでexitしたトークン数

        Returns:
            exit_counts, shallow_ratio, compute_costを含む辞書
        """
        total_tokens = sum(exit_counts)

        # 重み付きレイヤーコストを計算
        total_layers_computed = 0
        layers_so_far = 0
        for llm_idx, count in enumerate(exit_counts):
            layers_so_far += self.llms[llm_idx].num_layers
            total_layers_computed += count * layers_so_far

        compute_cost = (
            total_layers_computed / (total_tokens * self.num_layers)
            if total_tokens > 0 else 1.0
        )

        # Shallow ratio: 最終LLM以外でのexit
        shallow_exits = sum(exit_counts[:-1]) if len(exit_counts) > 1 else 0
        shallow_ratio = shallow_exits / total_tokens if total_tokens > 0 else 0.0

        return {
            'exit_counts': exit_counts,
            'shallow_ratio': shallow_ratio,
            'compute_cost': compute_cost,
        }
