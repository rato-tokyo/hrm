"""
CASCADEフレームワーク - 三角形Attention検出

Attention mapから意味的spanを検出する。
行方向の変化点検出による簡略化実装。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Span:
    """検出されたspanを表すデータクラス。"""
    start: int  # span開始位置（inclusive）
    end: int    # span終了位置（inclusive）
    score: float  # スコア（変化点検出では使用しない）

    @property
    def length(self) -> int:
        """spanの長さ。"""
        return self.end - self.start + 1

    def __repr__(self) -> str:
        return f"Span({self.start}-{self.end}, score={self.score:.3f})"


def detect_boundaries_by_row_change(
    attention_map: Tensor,
    threshold: float = 0.3,
) -> List[int]:
    """
    行方向のattention分布の変化点を検出。

    各行のattention分布が前の行と大きく異なる位置を境界とする。
    cos類似度が閾値以下の位置が境界。

    Args:
        attention_map: Attention weights (seq_len, seq_len)
                      Causal maskが適用済み（下三角のみ有効）
        threshold: cos類似度の閾値。これ以下で境界と判定。

    Returns:
        境界位置のリスト（開始位置0と終了位置seq_len-1を含む）
    """
    seq_len = attention_map.size(0)

    if seq_len <= 2:
        return [0, seq_len - 1]

    boundaries = [0]  # 開始位置は常に境界

    for i in range(1, seq_len):
        # 現在の行と前の行を比較
        # Causal maskのため、位置iの行は [0:i+1] の範囲のみ有効
        # 比較可能な範囲は [0:i] （前の行の有効範囲）
        if i < 2:
            # 比較に十分な長さがない場合はスキップ
            continue

        row_prev = attention_map[i - 1, :i]  # 前の行の有効部分
        row_curr = attention_map[i, :i]      # 現在の行の対応部分

        # ゼロベクトルチェック
        if row_prev.norm() < 1e-8 or row_curr.norm() < 1e-8:
            continue

        # cos類似度を計算
        cos_sim = F.cosine_similarity(
            row_prev.unsqueeze(0),
            row_curr.unsqueeze(0),
            dim=1
        ).item()

        # 類似度が閾値以下なら境界
        if cos_sim < threshold:
            boundaries.append(i)

    # 終了位置を追加（まだなければ）
    if boundaries[-1] != seq_len - 1:
        boundaries.append(seq_len - 1)

    return boundaries


def boundaries_to_spans(boundaries: List[int]) -> List[Span]:
    """
    境界位置リストをSpanリストに変換。

    Args:
        boundaries: 境界位置のリスト [0, 3, 7, 10] など

    Returns:
        Spanリスト [Span(0,3), Span(3,7), Span(7,10)] など
        ※ 境界位置は隣接spanで共有される
    """
    if len(boundaries) < 2:
        return []

    spans = []
    for i in range(len(boundaries) - 1):
        spans.append(Span(
            start=boundaries[i],
            end=boundaries[i + 1],
            score=1.0,  # 変化点検出ではスコアは使用しない
        ))

    return spans


def detect_span_boundaries(
    attention_map: Tensor,
    threshold: float = 0.3,
    iou_threshold: float = 0.5,  # 互換性のため残すが使用しない
    min_span_length: int = 1,
) -> Tuple[List[Span], Tensor]:
    """
    Attention mapからspan境界を検出。

    行方向のcos類似度変化点を境界として検出する簡略化版。

    Args:
        attention_map: Attention weights (seq_len, seq_len) または
                      (num_heads, seq_len, seq_len)
        threshold: cos類似度の閾値（低いほど境界検出が厳しい）
        iou_threshold: 未使用（互換性のため）
        min_span_length: 最小span長

    Returns:
        spans: 検出されたspanのリスト
        boundary_positions: 境界位置のテンソル
    """
    # 複数ヘッドの場合は平均
    if attention_map.dim() == 3:
        attention_map = attention_map.mean(dim=0)

    seq_len = attention_map.size(0)

    # 行方向の変化点を検出
    boundaries = detect_boundaries_by_row_change(attention_map, threshold)

    # 境界をspanに変換
    spans = boundaries_to_spans(boundaries)

    # 最小長でフィルタ（短すぎるspanをマージ）
    spans = _merge_short_spans(spans, min_span_length)

    # 全シーケンスをカバーするよう調整
    spans = _ensure_full_coverage(spans, seq_len)

    # 境界位置を抽出（各spanの最初と最後）
    boundary_positions = []
    for span in spans:
        boundary_positions.append(span.start)
        if span.end != span.start:
            boundary_positions.append(span.end)

    # 重複を除去してソート
    boundary_positions = sorted(set(boundary_positions))

    return spans, torch.tensor(boundary_positions, dtype=torch.long)


def _merge_short_spans(spans: List[Span], min_length: int) -> List[Span]:
    """短すぎるspanを前のspanにマージ。"""
    if not spans or min_length <= 1:
        return spans

    result: List[Span] = []
    for span in spans:
        if span.length < min_length and result:
            # 前のspanを拡張
            prev = result[-1]
            result[-1] = Span(
                start=prev.start,
                end=span.end,
                score=prev.score,
            )
        else:
            result.append(span)

    return result


def _ensure_full_coverage(spans: List[Span], seq_len: int) -> List[Span]:
    """
    全シーケンスをカバーするようにspanを調整。
    """
    if not spans:
        return [Span(start=0, end=seq_len - 1, score=1.0)]

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
