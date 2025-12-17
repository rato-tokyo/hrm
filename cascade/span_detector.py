"""
CASCADEフレームワーク - Span検出

Attention mapから意味的spanを検出するモジュール。

LTri-LLM論文ベースのTriangleScoreDetectorを使用。

Reference:
- LTri-LLM: Streaming Long Context Inference for LLMs with Training-Free
  Dynamic Triangular Attention Pattern (arXiv:2412.04757)
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Span:
    """検出されたspanを表すデータクラス。"""
    start: int  # span開始位置（inclusive）
    end: int    # span終了位置（inclusive）
    score: float  # spanのスコア（三角形スコア）

    @property
    def length(self) -> int:
        """spanの長さ。"""
        return self.end - self.start + 1

    def __repr__(self) -> str:
        return f"Span({self.start}-{self.end}, score={self.score:.3f})"


class TriangleScoreDetector:
    """
    LTri-LLM論文ベースの三角形スコアによるspan検出。

    三角形領域内のattention値の合計をスコアとし、
    NMS（Non-Maximum Suppression）で重複を除去。

    Reference: arXiv:2412.04757
    """

    def __init__(
        self,
        threshold: float = 0.0,
        iou_threshold: float = 0.3,
        min_span_length: int = 2,
        max_span_length: Optional[int] = 8,
    ):
        """
        Args:
            threshold: 三角形スコアの閾値（これ以下のスコアを負値に変換）
            iou_threshold: NMSのIoU閾値（デフォルト: 0.3）
            min_span_length: 最小span長
            max_span_length: 最大span長（デフォルト: 8）
        """
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length

    def detect(self, attention_map: Tensor) -> List[Span]:
        """
        三角形スコアとNMSによるspan検出。

        Args:
            attention_map: (seq_len, seq_len) または (num_heads, seq_len, seq_len)

        Returns:
            検出されたspanのリスト（スコア降順）
        """
        # 複数ヘッドの場合は合計（論文通り）
        if attention_map.dim() == 3:
            attention_map = attention_map.sum(dim=0)

        seq_len = attention_map.size(0)

        # 三角形スコアを計算
        triangle_scores = self._compute_triangle_scores(attention_map)

        # 候補span生成
        candidates = self._generate_candidates(triangle_scores, seq_len)

        if not candidates:
            # 候補がない場合は全体を1つのspanに
            return [Span(start=0, end=seq_len - 1, score=0.0)]

        # NMSで重複除去
        spans = self._apply_nms(candidates)

        # 全シーケンスをカバーするよう調整
        spans = self._ensure_full_coverage(spans, seq_len)

        return spans

    def _compute_triangle_scores(self, attention_map: Tensor) -> Tensor:
        """
        三角形スコアを計算。

        LTri-LLM論文の式:
        S_xy = Σ_i(x→y) Σ_j(x→y) (A_ij - θ)

        累積和を使用してO(n²)で全spanスコアを計算。

        Args:
            attention_map: (seq_len, seq_len)

        Returns:
            triangle_scores: (seq_len, seq_len) where [i,j] = span(i,j)のスコア
        """
        seq_len = attention_map.size(0)

        # 閾値を適用（論文のbaseline reward）
        thresholded = attention_map - self.threshold

        # 累積和で効率的に計算
        cumsum = torch.zeros(seq_len + 1, seq_len + 1, device=attention_map.device)
        cumsum[1:, 1:] = torch.cumsum(torch.cumsum(thresholded, dim=0), dim=1)

        # 三角形スコア行列を計算
        triangle_scores = torch.zeros(seq_len, seq_len, device=attention_map.device)

        for x in range(seq_len):
            for y in range(x, seq_len):
                score = (cumsum[y + 1, y + 1] - cumsum[x, y + 1]
                         - cumsum[y + 1, x] + cumsum[x, x])
                triangle_scores[x, y] = score

        return triangle_scores

    def _generate_candidates(self, triangle_scores: Tensor, seq_len: int) -> List[Span]:
        """候補spanを生成。"""
        candidates = []
        max_len = self.max_span_length or seq_len

        for x in range(seq_len):
            for y in range(x + self.min_span_length - 1, min(x + max_len, seq_len)):
                score = triangle_scores[x, y].item()
                if score > 0:
                    candidates.append(Span(start=x, end=y, score=score))

        candidates.sort(key=lambda s: s.score, reverse=True)
        return candidates

    def _apply_nms(self, candidates: List[Span]) -> List[Span]:
        """NMS（Non-Maximum Suppression）で重複除去。"""
        if not candidates:
            return []

        selected: List[Span] = []

        for candidate in candidates:
            is_suppressed = False
            for selected_span in selected:
                iou = self._compute_iou(candidate, selected_span)
                if iou > self.iou_threshold:
                    is_suppressed = True
                    break

            if not is_suppressed:
                selected.append(candidate)

        return selected

    def _compute_iou(self, span1: Span, span2: Span) -> float:
        """2つのspanのIoU（Intersection over Union）を計算。"""
        inter_start = max(span1.start, span2.start)
        inter_end = min(span1.end, span2.end)
        intersection = max(0, inter_end - inter_start + 1)
        union = span1.length + span2.length - intersection
        if union == 0:
            return 0.0
        return intersection / union

    def _ensure_full_coverage(self, spans: List[Span], seq_len: int) -> List[Span]:
        """全シーケンスをカバーするようにspanを調整。"""
        if not spans:
            return [Span(start=0, end=seq_len - 1, score=0.0)]

        spans = sorted(spans, key=lambda x: x.start)
        result: List[Span] = []
        covered_until = -1

        for span in spans:
            if span.start > covered_until + 1:
                result.append(Span(
                    start=covered_until + 1,
                    end=span.start - 1,
                    score=0.0
                ))
            if span.end > covered_until:
                result.append(span)
                covered_until = span.end

        if covered_until < seq_len - 1:
            result.append(Span(
                start=covered_until + 1,
                end=seq_len - 1,
                score=0.0
            ))

        return result


# ユーティリティ関数
def spans_to_boundaries(spans: List[Span]) -> List[int]:
    """SpanリストからユニークなboundaryリストをI取得。"""
    boundaries = set()
    for span in spans:
        boundaries.add(span.start)
        boundaries.add(span.end)
    return sorted(boundaries)


def aggregate_attention_maps(
    attention_maps: Tuple[Tensor, ...],
    method: str = "mean",
    layer_weights: Optional[List[float]] = None,
) -> Tensor:
    """
    複数レイヤーのattention mapを集約。

    Args:
        attention_maps: 各レイヤーのattention (num_layers個の (batch, num_heads, seq, seq))
        method: 集約方法 ("mean" or "sum")
        layer_weights: レイヤーごとの重み

    Returns:
        集約されたattention map (batch, seq, seq)
    """
    valid_maps = [m for m in attention_maps if m is not None]
    if not valid_maps:
        raise ValueError("No valid attention maps")

    stacked = torch.stack(valid_maps, dim=0)

    if layer_weights is not None:
        weights = torch.tensor(layer_weights, device=stacked.device, dtype=stacked.dtype)
        weights = weights.view(-1, 1, 1, 1, 1)
        stacked = stacked * weights

    if method == "sum":
        aggregated = stacked.sum(dim=(0, 2))
    else:
        aggregated = stacked.mean(dim=(0, 2))

    return aggregated
