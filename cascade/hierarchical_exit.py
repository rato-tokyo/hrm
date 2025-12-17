"""
CASCADEフレームワーク - 階層的Early Exit

三角形Attentionとspan圧縮を統合し、
多段階で情報を圧縮していく処理を実現。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from .span_detector import (
    Span,
    TriangleScoreDetector,
    aggregate_attention_maps,
)
from .span_compressor import (
    SpanCompressor,
    extract_span_boundaries,
)

if TYPE_CHECKING:
    from .llm import LLM


@dataclass
class StageOutput:
    """各段階の出力を表すデータクラス。"""
    hidden_states: Tensor  # 出力hidden states
    spans: List[Span]  # 検出されたspan
    boundary_positions: Tensor  # 境界位置（元のシーケンスでの位置）
    attention_map: Optional[Tensor] = None  # Attention map（デバッグ用）


@dataclass
class HierarchicalOutput:
    """階層的処理の全体出力。"""
    final_hidden: Tensor  # 最終出力hidden states
    stage_outputs: List[StageOutput] = field(default_factory=list)
    compression_ratios: List[float] = field(default_factory=list)

    @property
    def total_compression_ratio(self) -> float:
        """全体の圧縮率。"""
        if not self.compression_ratios:
            return 0.0
        ratio = 1.0
        for r in self.compression_ratios:
            ratio *= (1.0 - r)
        return 1.0 - ratio


class HierarchicalExit(nn.Module):
    """
    階層的Early Exitモジュール。

    複数のLLMを用いて、三角形Attentionでspan境界を検出し、
    段階的に情報を圧縮していく。

    アーキテクチャ:
    ```
    入力: [t0, t1, t2, ..., tN]  (N tokens)
              ↓
    LLM 0: 全トークン処理 + Attention取得
              ↓
    三角形検出: span境界を抽出
              ↓
    出力: [h_b0, h_e0, h_b1, h_e1, ...]  (境界のみ、約 N/k tokens)
              ↓
    LLM 1: 境界トークン処理 + Attention取得
              ↓
    三角形検出: さらに境界を抽出
              ↓
    出力: [h'_b0, h'_e0, ...]  (さらに圧縮、約 N/k^2 tokens)
              ↓
           ...
    ```

    使用例:
        from transformers import AutoModelForCausalLM
        from cascade import LLM

        llm_0 = LLM(AutoModelForCausalLM.from_pretrained('gpt2'))
        llm_1 = LLM(AutoModelForCausalLM.from_pretrained('gpt2'))

        hierarchical = HierarchicalExit([llm_0, llm_1])
        output = hierarchical(token_ids)

        print(f"圧縮率: {output.total_compression_ratio:.1%}")
    """

    def __init__(
        self,
        llms: List["LLM"],
        span_detector: Optional[TriangleScoreDetector] = None,
        min_output_length: int = 2,
    ):
        """
        Args:
            llms: LLMのリスト（段階数 = len(llms)）
            span_detector: Span検出器（Noneの場合はTriangleScoreDetectorを使用）
            min_output_length: 最小出力長（これ以下には圧縮しない）
        """
        super().__init__()
        self.llms = nn.ModuleList(llms)
        self.span_detector = span_detector or TriangleScoreDetector()
        self.compressor = SpanCompressor(min_output_length=min_output_length)

    @property
    def num_stages(self) -> int:
        """段階数。"""
        return len(self.llms)

    def forward(
        self,
        token_ids: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
        return_all_stages: bool = False,
    ) -> HierarchicalOutput:
        """
        階層的処理を実行。

        Args:
            token_ids: 入力トークンID (batch, seq_len)。最初の段階用。
            hidden_states: 入力hidden states (batch, seq_len, dim)。
                          token_idsが指定されている場合は無視。
            return_all_stages: 全段階の出力を返すか

        Returns:
            HierarchicalOutput
        """
        if token_ids is None and hidden_states is None:
            raise ValueError("token_ids または hidden_states のどちらかを指定してください")

        stage_outputs: List[StageOutput] = []
        compression_ratios: List[float] = []

        current_hidden = hidden_states
        current_positions: Optional[Tensor] = None

        for i, llm in enumerate(self.llms):
            # LLM処理
            if i == 0 and token_ids is not None:
                # 最初の段階: token_ids入力
                h_out, attentions = self._forward_with_attention(llm, token_ids=token_ids)
                original_seq_len = token_ids.size(1)
                current_positions = torch.arange(
                    original_seq_len,
                    device=token_ids.device
                )
            else:
                # 後続段階: hidden_states入力
                assert current_hidden is not None
                h_out, attentions = self._forward_with_attention(
                    llm, hidden_states=current_hidden
                )
                original_seq_len = current_hidden.size(1)

            # Attention mapを集約してspan検出
            if attentions is not None:
                agg_attention = aggregate_attention_maps(attentions)
                attention_for_detection = agg_attention[0]  # バッチの最初
                spans = self.span_detector.detect(attention_for_detection)
            else:
                # Attentionが取得できない場合はダミーのattention mapで検出
                dummy_attention = torch.zeros(h_out.size(1), h_out.size(1), device=h_out.device)
                spans = self.span_detector.detect(dummy_attention)

            # span境界のhidden statesを抽出
            boundary_hidden, local_positions = extract_span_boundaries(h_out, spans)

            # 元のシーケンスでの位置を追跡
            if current_positions is not None:
                global_positions = current_positions[local_positions.to(current_positions.device)]
            else:
                global_positions = local_positions

            # 圧縮率を計算
            input_len = h_out.size(1)
            output_len = boundary_hidden.size(1)
            ratio = 1.0 - (output_len / input_len) if input_len > 0 else 0.0
            compression_ratios.append(ratio)

            # 段階出力を保存
            stage_output = StageOutput(
                hidden_states=boundary_hidden,
                spans=spans,
                boundary_positions=global_positions,
                attention_map=agg_attention if return_all_stages else None,
            )
            stage_outputs.append(stage_output)

            # 次の段階への入力を更新
            current_hidden = boundary_hidden
            current_positions = global_positions

        # current_hiddenは必ず存在（llmsが空でない限り）
        assert current_hidden is not None
        return HierarchicalOutput(
            final_hidden=current_hidden,
            stage_outputs=stage_outputs if return_all_stages else [],
            compression_ratios=compression_ratios,
        )

    def _forward_with_attention(
        self,
        llm: "LLM",
        token_ids: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]]]:
        """
        LLMをforward実行し、attention mapも取得。

        Args:
            llm: LLMインスタンス
            token_ids: 入力トークンID
            hidden_states: 入力hidden states

        Returns:
            h_out: 出力hidden states
            attentions: Attention weights（取得できない場合はNone）
        """
        try:
            if token_ids is not None:
                outputs = llm.base_llm(
                    input_ids=token_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                )
            else:
                assert hidden_states is not None
                # dtypeを合わせる
                model_dtype = next(llm.base_llm.parameters()).dtype
                if hidden_states.dtype != model_dtype:
                    hidden_states = hidden_states.to(dtype=model_dtype)

                outputs = llm.base_llm(
                    inputs_embeds=hidden_states,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                )

            h_out = outputs.hidden_states[-1]
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None

            return h_out, attentions

        except Exception:
            # Attention取得に失敗した場合はhidden_statesのみ返す
            if token_ids is not None:
                h_out, _ = llm.forward_token_ids(token_ids)
            else:
                assert hidden_states is not None
                h_out, _ = llm.forward_hidden_states(hidden_states)
            return h_out, None

    def get_compression_stats(self, output: HierarchicalOutput) -> dict:
        """圧縮統計を取得。"""
        return {
            "num_stages": self.num_stages,
            "compression_ratios": output.compression_ratios,
            "total_compression_ratio": output.total_compression_ratio,
            "final_length": output.final_hidden.size(1),
        }


def create_hierarchical_exit(
    llms: List["LLM"],
    **kwargs,
) -> HierarchicalExit:
    """
    HierarchicalExitを作成するファクトリ関数。

    Args:
        llms: LLMのリスト
        **kwargs: HierarchicalExitの追加引数

    Returns:
        HierarchicalExitインスタンス
    """
    return HierarchicalExit(llms, **kwargs)
