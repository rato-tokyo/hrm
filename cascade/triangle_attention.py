"""
CASCADEフレームワーク - 三角形Attention検出

Attention mapから三角形パターンを検出し、意味的spanを抽出する。
Ltri-LLMの手法に基づく実装。
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
    score: float  # 三角形attention score

    @property
    def length(self) -> int:
        """spanの長さ。"""
        return self.end - self.start + 1

    def __repr__(self) -> str:
        return f"Span({self.start}-{self.end}, score={self.score:.3f})"


def compute_triangle_scores(
    attention_map: Tensor,
    threshold: float = 0.0,
) -> Tensor:
    """
    三角形領域のattention scoreを計算。

    各位置(i, j)に対して、その位置を右下角とする三角形領域の
    累積attention scoreを計算する。

    Args:
        attention_map: Attention weights (seq_len, seq_len)
                      Causal maskが適用済み（上三角は0）
        threshold: スコア計算時に減算する閾値

    Returns:
        triangle_scores: 各位置の三角形score (seq_len, seq_len)
    """
    seq_len = attention_map.size(0)
    device = attention_map.device

    # 閾値を適用
    thresholded = attention_map - threshold
    thresholded = torch.clamp(thresholded, min=0.0)

    # 累積和で効率的に計算
    # cumsum_row[i, j] = sum(attention[i, 0:j+1])
    cumsum_row = torch.cumsum(thresholded, dim=1)

    # cumsum_both[i, j] = sum(attention[0:i+1, 0:j+1])
    cumsum_both = torch.cumsum(cumsum_row, dim=0)

    # 三角形スコアの計算
    # S(x, y) = 位置(x, x)から(y, y)までの三角形領域の累積
    triangle_scores = torch.zeros(seq_len, seq_len, device=device)

    for y in range(seq_len):
        for x in range(y + 1):
            # 三角形領域の累積を計算
            # (x, x) から (y, y) の三角形
            if x == 0:
                triangle_scores[x, y] = cumsum_both[y, y]
            else:
                # 上の長方形を引く
                triangle_scores[x, y] = cumsum_both[y, y] - cumsum_both[x - 1, y]
                # 左の長方形を引く（ただし三角形なので調整が必要）
                if x > 0:
                    # x列より左の部分を引く
                    triangle_scores[x, y] -= cumsum_both[y, x - 1]
                    if x > 1:
                        # 二重に引いた部分を足し戻す
                        triangle_scores[x, y] += cumsum_both[x - 1, x - 1]

    return triangle_scores


def compute_triangle_scores_efficient(
    attention_map: Tensor,
    threshold: float = 0.0,
) -> Tensor:
    """
    三角形領域のattention scoreを効率的に計算。

    対角線に沿ってスコアを計算し、各span候補のスコアを得る。

    Args:
        attention_map: Attention weights (seq_len, seq_len)
        threshold: スコア計算時に減算する閾値

    Returns:
        triangle_scores: 各span (start, end) のscore (seq_len, seq_len)
                        triangle_scores[start, end] = span (start, end) のスコア
    """
    seq_len = attention_map.size(0)
    device = attention_map.device
    dtype = attention_map.dtype

    # 閾値を適用
    thresholded = attention_map - threshold
    thresholded = torch.clamp(thresholded, min=0.0)

    # 結果テンソル
    triangle_scores = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

    # 対角線上のスコア（長さ1のspan）
    for i in range(seq_len):
        triangle_scores[i, i] = thresholded[i, i]

    # 長さ2以上のspan
    for length in range(2, seq_len + 1):
        for start in range(seq_len - length + 1):
            end = start + length - 1
            # 前のspanのスコア + 新しい行の寄与
            prev_score = triangle_scores[start, end - 1]
            # 新しい行(end)の、start から end までの合計
            new_row_sum = thresholded[end, start:end + 1].sum()
            triangle_scores[start, end] = prev_score + new_row_sum

    return triangle_scores


def detect_spans_nms(
    triangle_scores: Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    max_spans: Optional[int] = None,
) -> List[Span]:
    """
    NMS (Non-Maximum Suppression) でspanを検出。

    Args:
        triangle_scores: 三角形スコア (seq_len, seq_len)
        iou_threshold: IoU（重複度）の閾値。これ以上重複するspanは除去
        score_threshold: スコアの最小閾値
        max_spans: 最大span数（Noneの場合は制限なし）

    Returns:
        検出されたspanのリスト（スコア降順）
    """
    seq_len = triangle_scores.size(0)

    # 全span候補を収集
    candidates = []
    for start in range(seq_len):
        for end in range(start, seq_len):
            score = triangle_scores[start, end].item()
            if score > score_threshold:
                candidates.append(Span(start=start, end=end, score=score))

    if not candidates:
        return []

    # スコア降順でソート
    candidates.sort(key=lambda x: x.score, reverse=True)

    # NMS
    selected: List[Span] = []
    suppressed = set()

    for i, span in enumerate(candidates):
        if i in suppressed:
            continue

        selected.append(span)

        if max_spans is not None and len(selected) >= max_spans:
            break

        # 重複するspanを抑制
        for j in range(i + 1, len(candidates)):
            if j in suppressed:
                continue
            other = candidates[j]
            iou = _compute_span_iou(span, other)
            if iou > iou_threshold:
                suppressed.add(j)

    return selected


def _compute_span_iou(span1: Span, span2: Span) -> float:
    """2つのspanのIoU (Intersection over Union) を計算。"""
    # 交差部分
    inter_start = max(span1.start, span2.start)
    inter_end = min(span1.end, span2.end)

    if inter_start > inter_end:
        return 0.0

    intersection = inter_end - inter_start + 1

    # 和集合
    union = span1.length + span2.length - intersection

    return intersection / union if union > 0 else 0.0


def detect_span_boundaries(
    attention_map: Tensor,
    threshold: float = 0.0,
    iou_threshold: float = 0.5,
    min_span_length: int = 1,
) -> Tuple[List[Span], Tensor]:
    """
    Attention mapからspan境界を検出。

    Args:
        attention_map: Attention weights (seq_len, seq_len) または
                      (num_heads, seq_len, seq_len)
        threshold: 三角形スコアの閾値
        iou_threshold: NMSのIoU閾値
        min_span_length: 最小span長

    Returns:
        spans: 検出されたspanのリスト
        boundary_positions: 境界位置のテンソル [start0, end0, start1, end1, ...]
    """
    # 複数ヘッドの場合は平均
    if attention_map.dim() == 3:
        attention_map = attention_map.mean(dim=0)

    # 三角形スコア計算
    triangle_scores = compute_triangle_scores_efficient(attention_map, threshold)

    # NMSでspan検出
    spans = detect_spans_nms(
        triangle_scores,
        iou_threshold=iou_threshold,
        score_threshold=threshold,
    )

    # 最小長でフィルタ
    spans = [s for s in spans if s.length >= min_span_length]

    # 位置順でソート
    spans.sort(key=lambda x: x.start)

    # 重複を除去しつつカバレッジを確保
    spans = _ensure_full_coverage(spans, attention_map.size(0))

    # 境界位置を抽出
    boundary_positions = []
    for span in spans:
        boundary_positions.extend([span.start, span.end])

    # 重複を除去してソート
    boundary_positions = sorted(set(boundary_positions))

    return spans, torch.tensor(boundary_positions, dtype=torch.long)


def _ensure_full_coverage(spans: List[Span], seq_len: int) -> List[Span]:
    """
    全シーケンスをカバーするようにspanを調整。

    重複がある場合はスコアの高い方を優先し、
    カバーされていない部分は新しいspanを追加。
    """
    if not spans:
        # spanが検出されなかった場合、全体を1つのspanとする
        return [Span(start=0, end=seq_len - 1, score=0.0)]

    # 位置順でソート
    spans = sorted(spans, key=lambda x: x.start)

    result: List[Span] = []
    covered_until = -1

    for span in spans:
        if span.start > covered_until + 1:
            # ギャップがある場合、埋めるspanを追加
            result.append(Span(
                start=covered_until + 1,
                end=span.start - 1,
                score=0.0
            ))
        if span.end > covered_until:
            result.append(span)
            covered_until = span.end

    # 末尾のギャップを埋める
    if covered_until < seq_len - 1:
        result.append(Span(
            start=covered_until + 1,
            end=seq_len - 1,
            score=0.0
        ))

    return result


def aggregate_attention_maps(
    attention_maps: Tuple[Tensor, ...],
    layer_weights: Optional[List[float]] = None,
) -> Tensor:
    """
    複数レイヤーのattention mapを集約。

    Args:
        attention_maps: 各レイヤーのattention (num_layers, batch, num_heads, seq, seq)
        layer_weights: レイヤーごとの重み（Noneの場合は均等）

    Returns:
        集約されたattention map (batch, seq, seq)
    """
    # タプルをスタック
    stacked = torch.stack(attention_maps, dim=0)  # (num_layers, batch, num_heads, seq, seq)

    if layer_weights is not None:
        weights = torch.tensor(layer_weights, device=stacked.device, dtype=stacked.dtype)
        weights = weights.view(-1, 1, 1, 1, 1)
        stacked = stacked * weights

    # レイヤーとヘッドで平均
    aggregated = stacked.mean(dim=(0, 2))  # (batch, seq, seq)

    return aggregated
