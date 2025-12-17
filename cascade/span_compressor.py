"""
CASCADEフレームワーク - Span圧縮

検出されたspanから境界hidden statesを抽出し、
多段階で圧縮していくロジック。
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .span_detector import Span


@dataclass
class CompressedOutput:
    """圧縮結果を表すデータクラス。"""
    hidden_states: Tensor  # 圧縮後のhidden states (batch, num_boundaries, dim)
    boundary_positions: Tensor  # 元のシーケンスでの境界位置
    spans: List[Span]  # 検出されたspan情報
    original_seq_len: int  # 元のシーケンス長


def extract_span_boundaries(
    hidden_states: Tensor,
    spans: List[Span],
) -> Tuple[Tensor, Tensor]:
    """
    spanの最初と最後のhidden statesを抽出。

    Args:
        hidden_states: (batch, seq_len, dim)
        spans: 検出されたspanリスト

    Returns:
        boundary_hidden: 境界位置のhidden states (batch, num_boundaries, dim)
        boundary_positions: 境界位置のインデックス (num_boundaries,)
    """
    if not spans:
        # spanがない場合は全体を返す
        return hidden_states, torch.arange(hidden_states.size(1))

    # 境界位置を収集（各spanの最初と最後）
    boundary_positions = []
    for span in spans:
        boundary_positions.append(span.start)
        if span.end != span.start:  # 長さ1のspanでない場合
            boundary_positions.append(span.end)

    # 重複を除去してソート
    boundary_positions = sorted(set(boundary_positions))
    boundary_positions_tensor = torch.tensor(
        boundary_positions,
        dtype=torch.long,
        device=hidden_states.device
    )

    # hidden statesを抽出
    boundary_hidden = hidden_states[:, boundary_positions_tensor, :]

    return boundary_hidden, boundary_positions_tensor


def map_positions_through_compression(
    original_positions: Tensor,
    boundary_positions: Tensor,
) -> Tensor:
    """
    圧縮後の位置にマッピング。

    元のシーケンスでの位置を、圧縮後のシーケンスでの位置に変換。

    Args:
        original_positions: 元の位置 (N,)
        boundary_positions: 境界位置 (num_boundaries,)

    Returns:
        mapped_positions: 圧縮後の位置 (N,)
    """
    # boundary_positionsでの各original_positionの位置を見つける
    mapped = torch.zeros_like(original_positions)

    for i, pos in enumerate(original_positions):
        # posがboundary_positionsのどの位置に対応するか
        mask = boundary_positions == pos
        if mask.any():
            mapped[i] = mask.nonzero(as_tuple=True)[0][0]
        else:
            # 境界位置にない場合は最も近い境界を使用
            distances = torch.abs(boundary_positions - pos)
            mapped[i] = distances.argmin()

    return mapped


class SpanCompressor:
    """
    多段階でspan境界を抽出・圧縮するクラス。

    使用例:
        compressor = SpanCompressor()

        # Stage 1: 全トークン → span境界
        compressed1 = compressor.compress(hidden_states, spans)

        # Stage 2: span境界 → さらに圧縮
        compressed2 = compressor.compress(compressed1.hidden_states, new_spans)
    """

    def __init__(
        self,
        min_output_length: int = 2,
    ):
        """
        Args:
            min_output_length: 最小出力長（これ以下には圧縮しない）
        """
        self.min_output_length = min_output_length

    def compress(
        self,
        hidden_states: Tensor,
        spans: List[Span],
        original_positions: Optional[Tensor] = None,
    ) -> CompressedOutput:
        """
        hidden statesをspan境界に圧縮。

        Args:
            hidden_states: (batch, seq_len, dim)
            spans: 検出されたspanリスト
            original_positions: 元のシーケンスでの位置（追跡用）

        Returns:
            CompressedOutput
        """
        batch_size, seq_len, dim = hidden_states.shape
        device = hidden_states.device

        if original_positions is None:
            original_positions = torch.arange(seq_len, device=device)

        # 最小長チェック
        if seq_len <= self.min_output_length:
            return CompressedOutput(
                hidden_states=hidden_states,
                boundary_positions=original_positions,
                spans=spans,
                original_seq_len=seq_len,
            )

        # span境界を抽出
        boundary_hidden, local_boundary_positions = extract_span_boundaries(
            hidden_states, spans
        )

        # 元のシーケンスでの位置を更新
        global_boundary_positions = original_positions[local_boundary_positions]

        return CompressedOutput(
            hidden_states=boundary_hidden,
            boundary_positions=global_boundary_positions,
            spans=spans,
            original_seq_len=seq_len,
        )

    def multi_stage_compress(
        self,
        hidden_states: Tensor,
        spans_per_stage: List[List[Span]],
    ) -> List[CompressedOutput]:
        """
        複数段階で圧縮を実行。

        Args:
            hidden_states: (batch, seq_len, dim)
            spans_per_stage: 各段階のspanリスト

        Returns:
            各段階のCompressedOutputリスト
        """
        outputs = []
        current_hidden = hidden_states
        current_positions = None

        for spans in spans_per_stage:
            compressed = self.compress(
                current_hidden,
                spans,
                current_positions,
            )
            outputs.append(compressed)
            current_hidden = compressed.hidden_states
            current_positions = compressed.boundary_positions

        return outputs


def compute_compression_ratio(
    original_length: int,
    compressed_length: int,
) -> float:
    """圧縮率を計算。"""
    if original_length == 0:
        return 0.0
    return 1.0 - (compressed_length / original_length)


def create_position_mapping(
    spans: List[Span],
    seq_len: int,
) -> Tensor:
    """
    各トークンがどのspan境界に対応するかのマッピングを作成。

    Args:
        spans: spanリスト
        seq_len: シーケンス長

    Returns:
        mapping: 各位置の対応する境界インデックス (seq_len,)
    """
    # 境界位置を収集
    boundary_positions = []
    for span in spans:
        boundary_positions.append(span.start)
        if span.end != span.start:
            boundary_positions.append(span.end)
    boundary_positions = sorted(set(boundary_positions))

    # 各位置を最も近い境界にマッピング
    mapping = torch.zeros(seq_len, dtype=torch.long)

    for i in range(seq_len):
        min_dist = float('inf')
        best_idx = 0
        for idx, bp in enumerate(boundary_positions):
            dist = abs(i - bp)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        mapping[i] = best_idx

    return mapping


def reconstruct_from_boundaries(
    boundary_hidden: Tensor,
    boundary_positions: Tensor,
    original_seq_len: int,
    mode: str = "nearest",
) -> Tensor:
    """
    境界hidden statesから元のシーケンス長に復元。

    Args:
        boundary_hidden: (batch, num_boundaries, dim)
        boundary_positions: 境界位置 (num_boundaries,)
        original_seq_len: 元のシーケンス長
        mode: "nearest" (最近傍) or "interpolate" (線形補間)

    Returns:
        reconstructed: (batch, original_seq_len, dim)
    """
    batch_size, num_boundaries, dim = boundary_hidden.shape
    device = boundary_hidden.device
    dtype = boundary_hidden.dtype

    reconstructed = torch.zeros(
        batch_size, original_seq_len, dim,
        device=device, dtype=dtype
    )

    if mode == "nearest":
        # 最近傍補間
        for i in range(original_seq_len):
            distances = torch.abs(boundary_positions - i)
            nearest_idx = distances.argmin()
            reconstructed[:, i, :] = boundary_hidden[:, nearest_idx, :]

    elif mode == "interpolate":
        # 線形補間
        for i in range(original_seq_len):
            # 前後の境界を見つける
            before_mask = boundary_positions <= i
            after_mask = boundary_positions >= i

            if before_mask.any() and after_mask.any():
                before_idx = before_mask.nonzero(as_tuple=True)[0][-1]
                after_idx = after_mask.nonzero(as_tuple=True)[0][0]

                if before_idx == after_idx:
                    reconstructed[:, i, :] = boundary_hidden[:, before_idx, :]
                else:
                    # 線形補間
                    before_pos = boundary_positions[before_idx].item()
                    after_pos = boundary_positions[after_idx].item()
                    t = (i - before_pos) / (after_pos - before_pos)
                    reconstructed[:, i, :] = (
                        (1 - t) * boundary_hidden[:, before_idx, :] +
                        t * boundary_hidden[:, after_idx, :]
                    )
            elif before_mask.any():
                before_idx = before_mask.nonzero(as_tuple=True)[0][-1]
                reconstructed[:, i, :] = boundary_hidden[:, before_idx, :]
            else:
                after_idx = after_mask.nonzero(as_tuple=True)[0][0]
                reconstructed[:, i, :] = boundary_hidden[:, after_idx, :]

    return reconstructed
